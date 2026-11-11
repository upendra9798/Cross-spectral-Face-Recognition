from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

from e3_d2_train import (
    CHECKPOINT,
    FEATURE_ROOT,
    OUTPUT_ROOT,
    HeadPair,
    objective_loss,
    load_features,
    load_initial_heads,
)
from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent
SIGMAS = (3.0, 7.0, 15.0)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def ranking_accuracy(uncertainty: torch.Tensor, distance: torch.Tensor, threshold: float) -> float:
    correct = 0
    total = 0
    for left in range(4):
        for right in range(left + 1, 4):
            delta_distance = distance[:, left] - distance[:, right]
            mask = delta_distance.abs() > threshold
            if mask.any():
                delta_uncertainty = uncertainty[mask, left] - uncertainty[mask, right]
                correct += int((delta_uncertainty * delta_distance[mask] > 0).sum())
                total += int(mask.sum())
    return float(correct / total) if total else float("nan")


def evaluate_clean(
    model: HeadPair,
    split: str,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    dataset = load_features(split)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    rgb_all, thermal_all, distance_all, combined_all, reliability_all = [], [], [], [], []
    with torch.no_grad():
        for rgb, thermal, distance in loader:
            rgb_u, thermal_u, combined = model(rgb.to(device), thermal.to(device))
            rgb_all.append(rgb_u.cpu())
            thermal_all.append(thermal_u.cpu())
            combined_all.append(combined.cpu())
            distance_all.append(distance)
            reliability_all.append(F.softmax(-combined.cpu(), dim=1))
    rgb = torch.cat(rgb_all)
    thermal = torch.cat(thermal_all)
    combined = torch.cat(combined_all)
    distance = torch.cat(distance_all)
    reliability = torch.cat(reliability_all)
    aggregate = {
        "split": split,
        "rgb_mean": float(rgb.mean()),
        "rgb_std": float(rgb.std(unbiased=False)),
        "thermal_mean": float(thermal.mean()),
        "thermal_std": float(thermal.std(unbiased=False)),
        "combined_mean": float(combined.mean()),
        "combined_std": float(combined.std(unbiased=False)),
        "reliability_std": float(reliability.std(unbiased=False)),
        "rgb_pearson": pearson(rgb.flatten(), distance.flatten()),
        "thermal_pearson": pearson(thermal.flatten(), distance.flatten()),
        "combined_pearson": pearson(combined.flatten(), distance.flatten()),
        "rgb_spearman": pearson(rgb.flatten().argsort().argsort().float(), distance.flatten().argsort().argsort().float()),
        "thermal_spearman": pearson(thermal.flatten().argsort().argsort().float(), distance.flatten().argsort().argsort().float()),
        "combined_spearman": pearson(combined.flatten().argsort().argsort().float(), distance.flatten().argsort().argsort().float()),
        "ranking_accuracy": ranking_accuracy(combined, distance, threshold),
        "clamp_hit_rate": 0.0,
        "zero_gradient_rate": float("nan"),
    }
    return aggregate, rgb, thermal, combined, distance, reliability


def blur_tensor(images: torch.Tensor, landmarks: torch.Tensor, region_index: int, sigma: float) -> torch.Tensor:
    extractor = RegionalFeatureExtractor()
    boxes = extractor.image_boxes(landmarks)
    blurred = TF.gaussian_blur(images, [9, 9], sigma=sigma)
    result = images.clone()
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[index, :, y1:y2, x1:x2] = blurred[index, :, y1:y2, x1:x2]
    return result


def evaluate_corruptions(
    heads: HeadPair,
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    teacher = load_pretrained_edgeface()
    adapter = EdgeFaceTransformerAdapter(copy.deepcopy(teacher)).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    adapter.load_state_dict(state["model_state_dict"], strict=True)
    adapter.rgb_uncertainty_head.load_state_dict(heads.rgb.state_dict())
    adapter.thermal_uncertainty_head.load_state_dict(heads.thermal.state_dict())
    adapter.eval()
    loader = DataLoader(SFTL54PairDataset(split="val"), batch_size=64, shuffle=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            clean_output = adapter.forward_with_uncertainty(
                batch["rgb"].to(device), batch["thermal"].to(device), batch["landmarks"].to(device)
            )
            clean_distance = 1.0 - F.cosine_similarity(
                clean_output["rgb_regional_features"],
                clean_output["thermal_regional_features"],
                dim=-1,
            ).cpu()
            for region_index, region in enumerate(REGION_NAMES):
                for sigma in SIGMAS:
                    rgb = blur_tensor(batch["rgb"], batch["landmarks"], region_index, sigma)
                    thermal = blur_tensor(batch["thermal"], batch["landmarks"], region_index, sigma)
                    output = adapter.forward_with_uncertainty(
                        rgb.to(device), thermal.to(device), batch["landmarks"].to(device)
                    )
                    distance = 1.0 - F.cosine_similarity(
                        output["rgb_regional_features"],
                        output["thermal_regional_features"],
                        dim=-1,
                    ).cpu()
                    for sample in range(distance.shape[0]):
                        rows.append({
                            "variant": "",
                            "pair_key": batch["pair_key"][sample],
                            "region": region,
                            "sigma": sigma,
                            "target_region_distance_change": float(distance[sample, region_index] - clean_distance[sample, region_index]),
                            "rgb_uncertainty_change": float(output["rgb_uncertainty"][sample, region_index].cpu() - clean_output["rgb_uncertainty"][sample, region_index].cpu()),
                            "thermal_uncertainty_change": float(output["thermal_uncertainty"][sample, region_index].cpu() - clean_output["thermal_uncertainty"][sample, region_index].cpu()),
                            "combined_uncertainty_change": float(output["combined_uncertainty"][sample, region_index].cpu() - clean_output["combined_uncertainty"][sample, region_index].cpu()),
                            "corrupted_distance": float(distance[sample, region_index]),
                        })
    return rows


def gradient_audit(
    heads: HeadPair,
    split: str,
    variant: str,
    threshold: float,
    device: torch.device,
) -> tuple[float, float]:
    dataset = load_features(split)
    rgb, thermal, distance = dataset.tensors
    rgb = rgb[:256].to(device)
    thermal = thermal[:256].to(device)
    distance = distance[:256].to(device)
    raw_rgb = heads.rgb.network(rgb).squeeze(-1)
    raw_thermal = heads.thermal.network(thermal).squeeze(-1)
    raw_rgb.retain_grad()
    raw_thermal.retain_grad()
    combined = raw_rgb.clamp(-4.0, 4.0) + raw_thermal.clamp(-4.0, 4.0)
    base_variant = "ranking" if variant.startswith("ranking") else (
        "combined" if variant.startswith("combined") else variant
    )
    loss, _ = objective_loss(base_variant, combined, distance, threshold, 1.0)
    gradients = torch.autograd.grad(loss, (raw_rgb, raw_thermal))
    zero_rate = float(
        torch.cat([(grad == 0).flatten() for grad in gradients]).float().mean()
    )
    clamp_rate = float(
        torch.cat(
            [
                ((raw_rgb <= -4.0) | (raw_rgb >= 4.0)).flatten(),
                ((raw_thermal <= -4.0) | (raw_thermal >= 4.0)).flatten(),
            ]
        ).float().mean()
    )
    return clamp_rate, zero_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D2 evaluation.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.000161975622)
    args = parser.parse_args()
    device = torch.device(args.device)
    model_variants = {}
    history = pd.read_csv(args.output / "e3_d2_training_history.csv")
    for checkpoint in sorted((args.output / "e3_d2_checkpoints").glob("*_heads.pt")):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        heads = HeadPair().to(device)
        heads.load_state_dict(payload["head_state_dict"])
        model_variants[payload["reported_variant"]] = heads.eval()
    results = []
    region_results = []
    corruption_results = []
    for variant, heads in model_variants.items():
        start = time.perf_counter()
        train_metrics, train_rgb, train_thermal, train_combined, train_distance, train_reliability = evaluate_clean(
            heads, "train", device, args.threshold
        )
        val_metrics, val_rgb, val_thermal, val_combined, val_distance, val_reliability = evaluate_clean(
            heads, "val", device, args.threshold
        )
        results.append({
            "variant": variant,
            "train_loss": float(history[history["reported_variant"] == variant]["train_loss"].iloc[-1]),
            "validation_loss": float(history[history["reported_variant"] == variant]["val_loss"].iloc[-1]),
            "train_rgb_mean": train_metrics["rgb_mean"],
            "train_rgb_std": train_metrics["rgb_std"],
            "train_thermal_mean": train_metrics["thermal_mean"],
            "train_thermal_std": train_metrics["thermal_std"],
            "train_combined_mean": train_metrics["combined_mean"],
            "train_combined_std": train_metrics["combined_std"],
            "validation_rgb_mean": val_metrics["rgb_mean"],
            "validation_rgb_std": val_metrics["rgb_std"],
            "validation_thermal_mean": val_metrics["thermal_mean"],
            "validation_thermal_std": val_metrics["thermal_std"],
            "validation_combined_mean": val_metrics["combined_mean"],
            "validation_combined_std": val_metrics["combined_std"],
            "validation_reliability_std": val_metrics["reliability_std"],
            "validation_combined_pearson": val_metrics["combined_pearson"],
            "validation_combined_spearman": val_metrics["combined_spearman"],
            "validation_ranking_accuracy": val_metrics["ranking_accuracy"],
            "clamp_hit_rate": gradient_audit(heads, "val", variant, args.threshold, device)[0],
            "zero_gradient_rate": gradient_audit(heads, "val", variant, args.threshold, device)[1],
            "memory_note": "GPU memory unavailable; evaluation ran on CPU.",
            "runtime_seconds": time.perf_counter() - start,
        })
        for region_index, region in enumerate(REGION_NAMES):
            region_results.append({
                "variant": variant,
                "region": region,
                "train_rgb_mean": float(train_rgb[:, region_index].mean()),
                "train_rgb_std": float(train_rgb[:, region_index].std(unbiased=False)),
                "train_thermal_mean": float(train_thermal[:, region_index].mean()),
                "train_thermal_std": float(train_thermal[:, region_index].std(unbiased=False)),
                "train_combined_mean": float(train_combined[:, region_index].mean()),
                "train_combined_std": float(train_combined[:, region_index].std(unbiased=False)),
                "validation_rgb_mean": float(val_rgb[:, region_index].mean()),
                "validation_rgb_std": float(val_rgb[:, region_index].std(unbiased=False)),
                "validation_thermal_mean": float(val_thermal[:, region_index].mean()),
                "validation_thermal_std": float(val_thermal[:, region_index].std(unbiased=False)),
                "validation_combined_mean": float(val_combined[:, region_index].mean()),
                "validation_combined_std": float(val_combined[:, region_index].std(unbiased=False)),
                "validation_distance_mean": float(val_distance[:, region_index].mean()),
                "validation_distance_std": float(val_distance[:, region_index].std(unbiased=False)),
                "validation_combined_pearson": pearson(val_combined[:, region_index], val_distance[:, region_index]),
                "validation_reliability_std": float(val_reliability[:, region_index].std(unbiased=False)),
            })
        rows = evaluate_corruptions(heads, device, args.threshold)
        for row in rows:
            row["variant"] = variant
        corruption_results.extend(rows)
    pd.DataFrame(results).to_csv(args.output / "e3_d2_results.csv", index=False)
    pd.DataFrame(region_results).to_csv(args.output / "e3_d2_region_results.csv", index=False)
    pd.DataFrame(corruption_results).to_csv(args.output / "e3_d2_corruption_results.csv", index=False)
    corruption = pd.DataFrame(corruption_results)
    corruption_summary = (
        corruption.groupby(["variant", "region", "sigma"], as_index=False)
        .agg(
            matching_distance_change=("target_region_distance_change", "mean"),
            rgb_uncertainty_change=("rgb_uncertainty_change", "mean"),
            thermal_uncertainty_change=("thermal_uncertainty_change", "mean"),
            combined_uncertainty_change=("combined_uncertainty_change", "mean"),
        )
    )
    corruption_summary.to_csv(args.output / "e3_d2_corruption_summary.csv", index=False)
    report_rows = []
    for variant in sorted(corruption["variant"].unique()):
        subset = corruption[corruption["variant"] == variant]
        for region in REGION_NAMES:
            region_subset = subset[subset["region"] == region]
            grouped = region_subset.groupby("sigma")["combined_uncertainty_change"].mean()
            monotonic = bool(all(grouped.iloc[i] <= grouped.iloc[i + 1] for i in range(len(grouped) - 1)))
            report_rows.append({
                "variant": variant,
                "region": region,
                "mean_combined_change": float(region_subset["combined_uncertainty_change"].mean()),
                "mean_distance_change": float(region_subset["target_region_distance_change"].mean()),
                "corruption_monotonicity": monotonic,
            })
    pd.DataFrame(report_rows).to_csv(args.output / "e3_d2_corruption_behavior.csv", index=False)
    report = [
        "# E3-D2 controlled uncertainty-head-only training",
        "",
        "All variants trained only the RGB and thermal uncertainty heads using "
        "identical frozen regional feature tensors, initialization, ordering, "
        "optimizer, learning rate, seed, and epoch budget.",
        "",
        "Evaluation results are in `e3_d2_results.csv`, per-region results are in "
        "`e3_d2_region_results.csv`, and held-out corruption results are in "
        "`e3_d2_corruption_results.csv`.",
        "",
        "No backbone, Transformer Adapter, regional extractor, original E3 checkpoint, "
        "or reliability-weighted alignment was modified.",
        "",
        "Corruption evaluation used validation images with sigma 3, 7, and 15. "
        "No corruption targets were used for training or model selection.",
        "Clamp-hit and zero-gradient rates are measured on the validation objective "
        "at the final head parameters. GPU memory was unavailable because this run "
        "used CPU.",
    ]
    (args.output / "e3_d2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
