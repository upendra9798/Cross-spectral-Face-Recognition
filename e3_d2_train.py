from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from e3_d_objectives import (
    calibration_regularization,
    combined_normalized_ranking,
    normalized_heteroscedastic_fixed,
    original_heteroscedastic,
    pairwise_ranking,
)
from uncertainty_head import RegionalUncertaintyHead


ROOT = Path(__file__).parent
FEATURE_ROOT = ROOT / "e3_d_objective_diagnostic" / "frozen_features"
OUTPUT_ROOT = ROOT / "e3_d_objective_diagnostic" / "e3_d2"
CHECKPOINT_ROOT = OUTPUT_ROOT / "e3_d2_checkpoints"
CHECKPOINT = ROOT / "Experiment_3_Uncertainty_best.pt"
TRAINING_SCALE = 0.00135366382334
THRESHOLDS = {
    "p25": 0.000070199370,
    "p50": 0.000161975622,
    "p75": 0.000305652618,
}


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


class HeadPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rgb = RegionalUncertaintyHead(feature_dim=100)
        self.thermal = RegionalUncertaintyHead(feature_dim=100)

    def forward(
        self, rgb: torch.Tensor, thermal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_uncertainty = self.rgb(rgb)
        thermal_uncertainty = self.thermal(thermal)
        return rgb_uncertainty, thermal_uncertainty, rgb_uncertainty + thermal_uncertainty


def load_features(split: str) -> TensorDataset:
    artifact = torch.load(
        FEATURE_ROOT / f"{split}_features.pt", map_location="cpu", weights_only=False
    )
    return TensorDataset(
        artifact["rgb_regional_features"].float(),
        artifact["thermal_regional_features"].float(),
        artifact["matching_distances"].float(),
    )


def load_initial_heads() -> tuple[HeadPair, str]:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"]
    heads = HeadPair()
    heads.rgb.load_state_dict(
        {key.removeprefix("rgb_uncertainty_head."): value for key, value in state.items() if key.startswith("rgb_uncertainty_head.")}
    )
    heads.thermal.load_state_dict(
        {key.removeprefix("thermal_uncertainty_head."): value for key, value in state.items() if key.startswith("thermal_uncertainty_head.")}
    )
    return heads, state_hash(heads.state_dict())


def objective_loss(
    variant: str,
    combined: torch.Tensor,
    distances: torch.Tensor,
    threshold: float,
    ranking_weight: float,
) -> tuple[torch.Tensor, int]:
    if variant == "original":
        return original_heteroscedastic(combined, distances), 0
    if variant == "normalized_fixed":
        loss, _ = normalized_heteroscedastic_fixed(combined, distances, TRAINING_SCALE)
        return loss, 0
    if variant == "ranking":
        details = pairwise_ranking(combined, distances, threshold)
        return details.loss, details.valid_pairs
    if variant == "combined":
        loss, details, _ = combined_normalized_ranking(
            combined, distances, TRAINING_SCALE, threshold, ranking_weight
        )
        return loss, details.valid_pairs
    if variant == "calibration":
        loss, _ = calibration_regularization(combined, distances)
        return loss, 0
    raise ValueError(f"Unknown variant: {variant}")


def run_epoch(
    model: HeadPair,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    variant: str,
    threshold: float,
    ranking_weight: float,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    valid_pairs = 0
    batches = 0
    for rgb, thermal, distances in loader:
        rgb, thermal, distances = rgb.to(device), thermal.to(device), distances.to(device)
        rgb_uncertainty, thermal_uncertainty, combined = model(rgb, thermal)
        loss, pairs = objective_loss(
            variant, combined, distances, threshold, ranking_weight
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach())
        valid_pairs += pairs
        batches += 1
    if batches == 0:
        raise ValueError("No batches were produced.")
    return {
        "loss": total_loss / batches,
        "valid_ranking_pairs": valid_pairs / batches,
    }


def train_variant(
    name: str,
    base_heads: HeadPair,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    threshold: float,
    ranking_weight: float,
    seed: int,
) -> tuple[HeadPair, list[dict[str, object]], dict[str, object]]:
    set_seed(seed)
    model = HeadPair().to(device)
    model.load_state_dict(base_heads.state_dict())
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, name, threshold, ranking_weight, device
        )
        val_metrics = run_epoch(
            model, val_loader, None, name, threshold, ranking_weight, device
        )
        history.append({
            "variant": name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_valid_ranking_pairs": train_metrics["valid_ranking_pairs"],
            "val_valid_ranking_pairs": val_metrics["valid_ranking_pairs"],
            "seconds": time.perf_counter() - start,
        })
    final_hash = state_hash(model.state_dict())
    return model, history, {
        "variant": name,
        "initial_head_hash": initial_hash,
        "final_head_hash": final_hash,
        "changed": initial_hash != final_hash,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "threshold": threshold,
        "ranking_weight": ranking_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D2 uncertainty-head-only training.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.epochs = 1
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        load_features("train"), batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        load_features("val"), batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    base_heads, base_hash = load_initial_heads()
    variants = [
        ("original", 0.0),
        ("normalized_fixed", 0.0),
        ("ranking_p25", THRESHOLDS["p25"]),
        ("ranking_p50", THRESHOLDS["p50"]),
        ("ranking_p75", THRESHOLDS["p75"]),
        ("combined_p25", THRESHOLDS["p25"]),
        ("combined_p50", THRESHOLDS["p50"]),
        ("combined_p75", THRESHOLDS["p75"]),
        ("calibration", 0.0),
    ]
    all_history = []
    records = []
    for name, threshold in variants:
        base_variant = "ranking" if name.startswith("ranking") else (
            "combined" if name.startswith("combined") else name
        )
        model, history, record = train_variant(
            base_variant,
            base_heads,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.learning_rate,
            threshold,
            args.ranking_weight,
            args.seed,
        )
        record["reported_variant"] = name
        record["initial_hash_matches_all"] = record["initial_head_hash"] == base_hash
        all_history.extend([{**row, "reported_variant": name} for row in history])
        torch.save(
            {
                "head_state_dict": model.state_dict(),
                "reported_variant": name,
                "objective_variant": base_variant,
                "threshold": threshold,
                "training_scale": TRAINING_SCALE,
                "seed": args.seed,
            },
            CHECKPOINT_ROOT / f"{name}_heads.pt",
        )
        records.append(record)
    with (OUTPUT_ROOT / "training_records.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_initial_head_hash": base_hash,
                "checkpoint": str(CHECKPOINT),
                "training_scale": TRAINING_SCALE,
                "thresholds": THRESHOLDS,
                "device": str(device),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "records": records,
            },
            handle,
            indent=2,
        )
    with (OUTPUT_ROOT / "e3_d2_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_history[0]))
        writer.writeheader()
        writer.writerows(all_history)
    print(json.dumps({"base_initial_head_hash": base_hash, "records": records}, indent=2))


if __name__ == "__main__":
    main()
