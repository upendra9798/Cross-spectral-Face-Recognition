from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from sftl54_cross_modal_attention import (
    EdgeFaceCrossModalAttention,
    configure_cross_modal_student,
)
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_xedgeface_train import (
    contrastive_modality_loss,
    extract_embeddings,
    load_pretrained_edgeface,
    retrieval_metrics,
    self_distillation_loss,
)


ROOT = Path(__file__).parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paired_forward_losses(
    teacher: torch.nn.Module,
    student: EdgeFaceCrossModalAttention,
    batch: dict[str, torch.Tensor],
    lambda_distillation: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb = batch["rgb"]
    thermal = batch["thermal"]
    subject_ids = batch["label"]
    student_rgb, student_thermal = student.forward_paired(rgb, thermal)
    student_rgb = functional.normalize(student_rgb, dim=1)
    student_thermal = functional.normalize(student_thermal, dim=1)
    with torch.no_grad():
        teacher_rgb = functional.normalize(teacher(rgb), dim=1)
    alignment = contrastive_modality_loss(
        student_rgb, student_thermal, subject_ids, margin
    )
    distillation = self_distillation_loss(teacher_rgb, student(rgb).detach())
    total = (1.0 - lambda_distillation) * alignment + lambda_distillation * distillation
    return total, {
        "total": total.item(),
        "alignment": alignment.item(),
        "distillation": distillation.item(),
    }


def run_epoch(
    teacher: torch.nn.Module,
    student: EdgeFaceCrossModalAttention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_distillation: float,
    margin: float,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    student.train(training)
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
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss, values = paired_forward_losses(
            teacher, student, batch, lambda_distillation, margin
        )
        if training:
            loss.backward()
            optimizer.step()
        for key, value in values.items():
            totals[key] += value
        batches += 1
    if batches == 0:
        raise ValueError("The DataLoader produced no batches.")
    return {key: value / batches for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment 2 Cross-Modal Attention.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-distillation", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "Experiment_2_Cross_Modal_Attention.pt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_pretrained_edgeface().to(device).eval()
    student = EdgeFaceCrossModalAttention(copy.deepcopy(teacher)).to(device)
    trainable_names = configure_cross_modal_student(student, include_stage0=True)
    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
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
    print("Total parameters:", sum(p.numel() for p in student.parameters()))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in student.parameters() if p.requires_grad),
    )
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
                    "experiment": "Experiment_2_Cross_Modal_Attention",
                },
                args.output,
            )
    print("Best checkpoint:", args.output)
    print("Best validation RGB->thermal Rank-1:", f"{best_rank1:.4f}")
