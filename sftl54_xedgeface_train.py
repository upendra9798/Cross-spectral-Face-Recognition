from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset


ROOT = Path(__file__).parent
EDGEFACE_REPO = ROOT / "otroshi-edgeface-ce86851"
CHECKPOINT_PATH = EDGEFACE_REPO / "checkpoints" / "edgeface_xs_gamma_06.pt"


def load_pretrained_edgeface() -> nn.Module:
    """Load the local pretrained EdgeFace checkpoint."""
    sys.path.insert(0, str(EDGEFACE_REPO))
    from backbones import get_model

    model = get_model("edgeface_xs_gamma_06")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def configure_student(model: nn.Module, include_stage0: bool = False) -> list[str]:
    """Freeze the student except for xEdgeFace's selected adaptation modules."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_names = []
    for name, parameter in model.named_parameters():
        is_layer_norm = "norm" in name.lower()
        is_stem = name.startswith("model.stem.")
        is_stage0 = name.startswith("model.stages.0.")
        if is_layer_norm or is_stem or (include_stage0 and is_stage0):
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def contrastive_modality_loss(
    rgb_embeddings: torch.Tensor,
    thermal_embeddings: torch.Tensor,
    subject_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Apply xEdgeFace cosine contrastive alignment with in-batch negatives."""
    similarities = rgb_embeddings @ thermal_embeddings.T
    positive_loss = 1.0 - similarities.diagonal()

    different_subjects = subject_ids[:, None] != subject_ids[None, :]
    negative_scores = similarities[different_subjects]
    if negative_scores.numel() == 0:
        return positive_loss.mean()

    negative_loss = functional.relu(negative_scores - margin)
    return 0.5 * (positive_loss.mean() + negative_loss.mean())


def self_distillation_loss(
    teacher_rgb_embeddings: torch.Tensor,
    student_rgb_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Preserve pretrained RGB behavior with cosine self-distillation."""
    return (1.0 - functional.cosine_similarity(
        teacher_rgb_embeddings,
        student_rgb_embeddings,
        dim=1,
    )).mean()


def forward_losses(
    teacher: nn.Module,
    student: nn.Module,
    batch: dict[str, torch.Tensor],
    lambda_distillation: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the paper's modality-alignment and RGB self-distillation losses."""
    rgb = batch["rgb"]
    thermal = batch["thermal"]
    subject_ids = batch["label"]

    student_rgb = functional.normalize(student(rgb), dim=1)
    student_thermal = functional.normalize(student(thermal), dim=1)
    with torch.no_grad():
        teacher_rgb = functional.normalize(teacher(rgb), dim=1)

    alignment = contrastive_modality_loss(student_rgb, student_thermal, subject_ids, margin)
    distillation = self_distillation_loss(teacher_rgb, student_rgb)
    total = (1.0 - lambda_distillation) * alignment + lambda_distillation * distillation
    return total, {
        "total": total.item(),
        "alignment": alignment.item(),
        "distillation": distillation.item(),
    }


def run_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_distillation: float,
    margin: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    student.train(is_training)
    teacher.eval()
    totals = {"total": 0.0, "alignment": 0.0, "distillation": 0.0}
    batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        loss, values = forward_losses(teacher, student, batch, lambda_distillation, margin)
        if is_training:
            loss.backward()
            optimizer.step()
        for key, value in values.items():
            totals[key] += value
        batches += 1

    if batches == 0:
        raise ValueError("The DataLoader produced no batches.")
    return {key: value / batches for key, value in totals.items()}


def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    """Extract normalized embeddings and subject IDs for retrieval evaluation."""
    embeddings_rgb = []
    embeddings_thermal = []
    subject_ids = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            embeddings_rgb.append(functional.normalize(model(rgb), dim=1).cpu())
            embeddings_thermal.append(functional.normalize(model(thermal), dim=1).cpu())
            subject_ids.append(batch["label"])
    return (
        torch.cat(embeddings_rgb),
        torch.cat(embeddings_thermal),
        torch.cat(subject_ids),
    )


def retrieval_metrics(rgb_embeddings, thermal_embeddings, rgb_ids, thermal_ids):
    """Compute bidirectional Rank-1 and Rank-5 identity retrieval."""
    similarity = rgb_embeddings @ thermal_embeddings.T
    rgb_ranking = similarity.argsort(dim=1, descending=True)
    thermal_ranking = similarity.T.argsort(dim=1, descending=True)
    rgb_to_thermal = thermal_ids[rgb_ranking]
    thermal_to_rgb = rgb_ids[thermal_ranking]
    rgb_targets = rgb_ids[:, None]
    thermal_targets = thermal_ids[:, None]

    return {
        "rgb_to_thermal_rank1": (rgb_to_thermal[:, :1] == rgb_targets).any(1).float().mean().item(),
        "rgb_to_thermal_rank5": (rgb_to_thermal[:, :5] == rgb_targets).any(1).float().mean().item(),
        "thermal_to_rgb_rank1": (thermal_to_rgb[:, :1] == thermal_targets).any(1).float().mean().item(),
        "thermal_to_rgb_rank5": (thermal_to_rgb[:, :5] == thermal_targets).any(1).float().mean().item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the xEdgeFace adaptation on SF-TL54.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-distillation", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--include-stage0", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "sftl54_xedgeface_best.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_pretrained_edgeface().to(device).eval()
    student = copy.deepcopy(teacher).to(device)
    trainable_names = configure_student(student, include_stage0=args.include_stage0)
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    train_loader = DataLoader(
        SFTL54PairDataset(split="train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SFTL54PairDataset(split="val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Device:", device)
    print("Trainable parameters:", sum(p.numel() for p in student.parameters() if p.requires_grad))
    print("Adaptation:", "LN + stem + Stage 0" if args.include_stage0 else "LN + stem")
    print("Trainable tensors:", len(trainable_names))

    best_rank1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            teacher, student, train_loader, optimizer, device,
            args.lambda_distillation, args.margin, args.max_train_batches,
        )
        val_metrics = run_epoch(
            teacher, student, val_loader, None, device,
            args.lambda_distillation, args.margin, args.max_val_batches,
        )
        val_rgb, val_thermal, val_ids = extract_embeddings(student, val_loader, device)
        retrieval = retrieval_metrics(val_rgb, val_thermal, val_ids, val_ids)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['total']:.4f} | "
            f"val loss {val_metrics['total']:.4f} | "
            f"val RGB->thermal R1 {retrieval['rgb_to_thermal_rank1']:.4f}"
        )
        if retrieval["rgb_to_thermal_rank1"] > best_rank1:
            best_rank1 = retrieval["rgb_to_thermal_rank1"]
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "epoch": epoch,
                    "val_retrieval": retrieval,
                    "adaptation": "LN + stem + Stage 0" if args.include_stage0 else "LN + stem",
                },
                args.output,
            )

    print("Best checkpoint:", args.output)
    print("Best validation RGB->thermal Rank-1:", f"{best_rank1:.4f}")
