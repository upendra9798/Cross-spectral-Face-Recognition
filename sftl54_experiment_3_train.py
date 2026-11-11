from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import (
    EdgeFaceTransformerAdapter,
    configure_transformer_student,
)
from sftl54_xedgeface_train import load_pretrained_edgeface
from sftl54_xedgeface_train import contrastive_modality_loss, self_distillation_loss
from uncertainty_head import uncertainty_matching_loss


ROOT = Path(__file__).parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    teacher, model, loader, optimizer, device, lambda_uncertainty, lambda_distillation,
    margin, max_batches,
):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "alignment": 0.0, "distillation": 0.0, "uncertainty": 0.0}
    batches = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        output = model.forward_with_uncertainty(
            batch["rgb"], batch["thermal"], batch["landmarks"]
        )
        distance = uncertainty_matching_loss(
            output["rgb_regional_features"],
            output["thermal_regional_features"],
            output["combined_uncertainty"],
        )
        rgb_global = F.normalize(output["global_rgb_embedding"], dim=1)
        thermal_global = F.normalize(output["global_thermal_embedding"], dim=1)
        alignment = contrastive_modality_loss(
            rgb_global, thermal_global, batch["label"], margin
        )
        with torch.no_grad():
            teacher_rgb = F.normalize(teacher(batch["rgb"]), dim=1)
        distillation = self_distillation_loss(teacher_rgb, rgb_global)
        loss = alignment + lambda_distillation * distillation + lambda_uncertainty * distance
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals["total"] += float(loss.detach())
        totals["alignment"] += float(alignment.detach())
        totals["distillation"] += float(distillation.detach())
        totals["uncertainty"] += float(distance.detach())
        batches += 1
    if batches == 0:
        raise ValueError("No batches were produced.")
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 uncertainty control experiment.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-uncertainty", type=float, default=1.0)
    parser.add_argument("--lambda-distillation", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "Experiment_3_Uncertainty_best.pt")
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT / "Experiment_3_Uncertainty_history.csv",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_pretrained_edgeface().to(device).eval()
    model = EdgeFaceTransformerAdapter(copy.deepcopy(teacher)).to(device)
    trainable_names = configure_transformer_student(model, include_stage0=True)
    for name, parameter in model.named_parameters():
        if name.startswith(("rgb_uncertainty_head.", "thermal_uncertainty_head.")):
            parameter.requires_grad = True
            trainable_names.append(name)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    train_loader = DataLoader(SFTL54PairDataset(split="train"), args.batch_size, shuffle=True)
    val_loader = DataLoader(SFTL54PairDataset(split="val"), args.batch_size, shuffle=False)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            teacher, model, train_loader, optimizer, device,
            args.lambda_uncertainty, args.lambda_distillation, args.margin,
            args.max_train_batches,
        )
        val_loss = run_epoch(
            teacher, model, val_loader, None, device,
            args.lambda_uncertainty, args.lambda_distillation, args.margin,
            args.max_val_batches,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_loss.items()})
        row.update({f"val_{key}": value for key, value in val_loss.items()})
        history.append(row)
        print(
            f"epoch={epoch} train_total={train_loss['total']:.6f} "
            f"train_alignment={train_loss['alignment']:.6f} "
            f"train_distillation={train_loss['distillation']:.6f} "
            f"train_uncertainty={train_loss['uncertainty']:.6f} "
            f"val_total={val_loss['total']:.6f} "
            f"val_uncertainty={val_loss['uncertainty']:.6f}"
        )
        if val_loss["uncertainty"] < best_val:
            best_val = val_loss["uncertainty"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_loss,
                    "trainable_names": trainable_names,
                    "experiment": "E3_uncertainty",
                },
                args.output,
            )
    with args.history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"saved best checkpoint {args.output}")
    print(f"saved loss history {args.history}")


if __name__ == "__main__":
    main()
