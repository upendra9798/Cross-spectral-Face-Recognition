from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import (
    EdgeFaceTransformerAdapter,
    configure_transformer_student,
)
from sftl54_xedgeface_train import (
    forward_losses,
    extract_embeddings,
    load_pretrained_edgeface,
    retrieval_metrics,
    run_epoch,
)


ROOT = Path(__file__).parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment 1 Transformer Adapter.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-distillation", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Experiment_1_Transformer_Adapter.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_pretrained_edgeface().to(device).eval()
    student = EdgeFaceTransformerAdapter(copy.deepcopy(teacher)).to(device)
    trainable_names = configure_transformer_student(student, include_stage0=True)
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

    total_parameters = sum(parameter.numel() for parameter in student.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in student.parameters() if parameter.requires_grad
    )
    print("Device:", device)
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)
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
                    "experiment": "Experiment_1_Transformer_Adapter",
                },
                args.output,
            )

    print("Best checkpoint:", args.output)
    print("Best validation RGB->thermal Rank-1:", f"{best_rank1:.4f}")
