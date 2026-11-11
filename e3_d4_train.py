from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from e3_d2_train import HeadPair
from e3_d_objectives import pairwise_ranking


ROOT = Path(__file__).parent
FEATURE_ROOT = ROOT / "e3_d_objective_diagnostic" / "frozen_features"
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_d4"
CHECKPOINT_ROOT = OUTPUT / "checkpoints"
REFERENCE = (
    ROOT
    / "e3_d_objective_diagnostic"
    / "e3_d2"
    / "e3_d2_checkpoints"
    / "ranking_p50_heads.pt"
)
THRESHOLD = 0.000161975622


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(split: str) -> TensorDataset:
    artifact = torch.load(
        FEATURE_ROOT / f"{split}_features.pt", map_location="cpu", weights_only=False
    )
    return TensorDataset(
        artifact["rgb_regional_features"].float(),
        artifact["thermal_regional_features"].float(),
        artifact["matching_distances"].float(),
    )


def load_reference() -> tuple[HeadPair, str]:
    payload = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    if payload["reported_variant"] != "ranking_p50":
        raise ValueError("Reference checkpoint is not ranking_p50.")
    if abs(float(payload["threshold"]) - THRESHOLD) > 1e-12:
        raise ValueError("Reference checkpoint threshold mismatch.")
    model = HeadPair()
    model.load_state_dict(payload["head_state_dict"], strict=True)
    return model, file_hash(REFERENCE)


def run_epoch(
    model: HeadPair,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    valid_pairs = 0
    batches = 0
    for rgb, thermal, distance in loader:
        rgb_u, thermal_u, combined = model(rgb, thermal)
        details = pairwise_ranking(combined, distance, THRESHOLD)
        if training:
            optimizer.zero_grad(set_to_none=True)
            details.loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("Non-finite uncertainty-head gradient.")
            optimizer.step()
        total_loss += float(details.loss.detach())
        valid_pairs += details.valid_pairs
        batches += 1
    return {
        "loss": total_loss / batches,
        "valid_ranking_pairs_per_batch": valid_pairs / batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D4 longer ranking-only training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("E3-D4 epoch budget must be between 1 and 10.")
    set_seed(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(load_split("train"), args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(load_split("val"), args.batch_size, shuffle=False, num_workers=0)
    model, reference_hash = load_reference()
    initial_state_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader, None)
        checkpoint = CHECKPOINT_ROOT / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "head_state_dict": model.state_dict(),
                "epoch": epoch,
                "objective": "ranking_only",
                "threshold": THRESHOLD,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "reference_checkpoint": str(REFERENCE),
                "reference_file_hash": reference_hash,
                "initial_state_hash": initial_state_hash,
            },
            checkpoint,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": val_metrics["loss"],
            "train_valid_ranking_pairs_per_batch": train_metrics["valid_ranking_pairs_per_batch"],
            "validation_valid_ranking_pairs_per_batch": val_metrics["valid_ranking_pairs_per_batch"],
            "checkpoint": str(checkpoint),
        })
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}"
        )
    final_hash = state_hash(model.state_dict())
    (OUTPUT / "e3_d4_training_state.json").write_text(
        json.dumps(
            {
                "reference": str(REFERENCE),
                "reference_file_hash": reference_hash,
                "initial_state_hash": initial_state_hash,
                "final_state_hash": final_hash,
                "threshold": THRESHOLD,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "reference_file_hash": reference_hash,
        "initial_state_hash": initial_state_hash,
        "final_state_hash": final_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
