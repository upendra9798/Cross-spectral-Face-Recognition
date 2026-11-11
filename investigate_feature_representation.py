from __future__ import annotations

import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface
from uncertainty_head import (
    RegionalUncertaintyHead,
    regional_reliability,
    uncertainty_matching_loss,
)


ROOT = Path(__file__).parent
RESOLUTIONS = {"7x7": (100, "stage2"), "14x14": (64, "stage1"), "28x28": (32, "stage0")}
STRENGTHS = (3.0, 7.0, 15.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResolutionRegionalProbe(nn.Module):
    def __init__(self, input_channels: int, resolution: str, output_dim: int = 100) -> None:
        super().__init__()
        self.resolution = resolution
        self.projection = nn.Conv2d(input_channels, output_dim, kernel_size=1)
        self.extractor = RegionalFeatureExtractor(feature_channels=output_dim)
        self.rgb_head = RegionalUncertaintyHead(feature_dim=output_dim)
        self.thermal_head = RegionalUncertaintyHead(feature_dim=output_dim)

    def forward(
        self,
        rgb_features: torch.Tensor,
        thermal_features: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rgb_regions = self.extractor(self.projection(rgb_features), landmarks)
        thermal_regions = self.extractor(self.projection(thermal_features), landmarks)
        rgb_uncertainty = self.rgb_head(rgb_regions)
        thermal_uncertainty = self.thermal_head(thermal_regions)
        combined, reliability = regional_reliability(rgb_uncertainty, thermal_uncertainty)
        return {
            "rgb_regional_features": rgb_regions,
            "thermal_regional_features": thermal_regions,
            "rgb_uncertainty": rgb_uncertainty,
            "thermal_uncertainty": thermal_uncertainty,
            "combined_uncertainty": combined,
            "regional_reliability": reliability,
        }


def backbone_features(
    backbone: nn.Module,
    images: torch.Tensor,
    resolution: str,
) -> torch.Tensor:
    model = backbone.model
    features = model.stem(images)
    features = model.stages[0](features)
    if resolution == "28x28":
        return features
    features = model.stages[1](features)
    if resolution == "14x14":
        return features
    features = model.stages[2](features)
    tokens = features.flatten(2).transpose(1, 2)
    return tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], features.shape[2], features.shape[3]
    )


def probe_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    return uncertainty_matching_loss(
        output["rgb_regional_features"],
        output["thermal_regional_features"],
        output["combined_uncertainty"],
    )


def train_probe(
    probe: ResolutionRegionalProbe,
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    history = []
    backbone.eval()
    for epoch in range(1, epochs + 1):
        probe.train()
        total = 0.0
        batches = 0
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.no_grad():
                rgb_features = backbone_features(backbone, batch["rgb"], probe.resolution)
                thermal_features = backbone_features(backbone, batch["thermal"], probe.resolution)
            output = probe(rgb_features, thermal_features, batch["landmarks"])
            loss = probe_loss(output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append({"epoch": epoch, "train_uncertainty": total / batches})
    return history


def blur_region(images: torch.Tensor, landmarks: torch.Tensor, index: int, sigma: float) -> torch.Tensor:
    boxes = RegionalFeatureExtractor().image_boxes(landmarks)
    blurred = TF.gaussian_blur(images, kernel_size=[9, 9], sigma=sigma)
    result = images.clone()
    for sample, box in enumerate(boxes[:, index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[sample, :, y1:y2, x1:x2] = blurred[sample, :, y1:y2, x1:x2]
    return result


def capture_output(
    probe: ResolutionRegionalProbe | None,
    adapter: EdgeFaceTransformerAdapter,
    backbone: nn.Module,
    batch: dict[str, object],
    resolution: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rgb = batch["rgb"].to(device)
    thermal = batch["thermal"].to(device)
    landmarks = batch["landmarks"].to(device)
    with torch.no_grad():
        if resolution == "7x7":
            output = adapter.forward_with_uncertainty(rgb, thermal, landmarks)
            return {
                "rgb_regional_features": output["rgb_regional_features"],
                "thermal_regional_features": output["thermal_regional_features"],
                "rgb_uncertainty": output["rgb_uncertainty"],
                "thermal_uncertainty": output["thermal_uncertainty"],
                "combined_uncertainty": output["combined_uncertainty"],
                "regional_reliability": output["regional_reliability"],
            }
        return probe(
            backbone_features(backbone, rgb, resolution),
            backbone_features(backbone, thermal, resolution),
            landmarks,
        )


def append_metrics(rows: list[dict[str, object]], batch: dict[str, object], output: dict[str, torch.Tensor], condition: str, sigma: float, resolution: str) -> None:
    distance = 1.0 - F.cosine_similarity(
        output["rgb_regional_features"], output["thermal_regional_features"], dim=-1
    )
    for sample, pair_key in enumerate(batch["pair_key"]):
        for index, region in enumerate(REGION_NAMES):
            rows.append({
                "resolution": resolution,
                "pair_key": pair_key,
                "subject_id": int(batch["subject_id"][sample]),
                "condition": condition,
                "sigma": sigma,
                "region": region,
                "matching_distance": float(distance[sample, index]),
                "rgb_uncertainty": float(output["rgb_uncertainty"][sample, index]),
                "thermal_uncertainty": float(output["thermal_uncertainty"][sample, index]),
                "combined_uncertainty": float(output["combined_uncertainty"][sample, index]),
                "reliability": float(output["regional_reliability"][sample, index]),
            })


def analyze_resolution(
    resolution: str,
    adapter: EdgeFaceTransformerAdapter,
    backbone: nn.Module,
    probe: ResolutionRegionalProbe | None,
    loader: DataLoader,
    device: torch.device,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    probe_or_adapter = probe if probe is not None else adapter
    if probe is not None:
        probe.eval()
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        clean = capture_output(probe, adapter, backbone, batch, resolution, device)
        append_metrics(rows, batch, clean, "clean", 0.0, resolution)
        for index, region in enumerate(REGION_NAMES):
            for sigma in STRENGTHS:
                corrupted = dict(batch)
                corrupted["rgb"] = blur_region(batch["rgb"], batch["landmarks"], index, sigma)
                corrupted["thermal"] = blur_region(batch["thermal"], batch["landmarks"], index, sigma)
                append_metrics(
                    rows,
                    batch,
                    capture_output(probe, adapter, backbone, corrupted, resolution, device),
                    region,
                    sigma,
                    resolution,
                )
    elapsed = time.perf_counter() - start
    report = pd.DataFrame(rows)
    resolution_dir = output_root / resolution
    resolution_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(resolution_dir / "metrics.csv", index=False)
    clean = report[report["condition"] == "clean"]
    stats = pd.DataFrame([{
        "resolution": resolution,
        "clean_distance_mean": clean["matching_distance"].mean(),
        "clean_distance_std": clean["matching_distance"].std(),
        "rgb_uncertainty_mean": clean["rgb_uncertainty"].mean(),
        "rgb_uncertainty_std": clean["rgb_uncertainty"].std(),
        "thermal_uncertainty_mean": clean["thermal_uncertainty"].mean(),
        "thermal_uncertainty_std": clean["thermal_uncertainty"].std(),
        "reliability_std": clean["reliability"].std(),
        "clean_rgb_distance_pearson": clean["rgb_uncertainty"].corr(clean["matching_distance"]),
        "clean_rgb_distance_spearman": clean["rgb_uncertainty"].corr(clean["matching_distance"], method="spearman"),
        "clean_thermal_distance_pearson": clean["thermal_uncertainty"].corr(clean["matching_distance"]),
        "clean_thermal_distance_spearman": clean["thermal_uncertainty"].corr(clean["matching_distance"], method="spearman"),
        "seconds": elapsed,
        "seconds_per_batch": elapsed / len(loader),
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else float("nan")),
        "trainable_probe_parameters": sum(p.numel() for p in probe_or_adapter.parameters() if p.requires_grad),
    }])
    stats.to_csv(resolution_dir / "statistics.csv", index=False)
    for region in REGION_NAMES:
        subset = report[report["region"] == region]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for condition, group in subset.groupby("condition", sort=False):
            means = group.groupby("sigma")[["combined_uncertainty", "matching_distance"]].mean()
            axes[0].plot(means.index, means["combined_uncertainty"], marker="o", label=condition)
            axes[1].plot(means.index, means["matching_distance"], marker="o", label=condition)
        axes[0].set_title(f"{resolution} {region} uncertainty")
        axes[1].set_title(f"{resolution} {region} distance")
        for axis in axes:
            axis.set_xlabel("blur sigma")
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(resolution_dir / f"{region}.png", dpi=150)
        plt.close(figure)
    return report, stats.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-C feature-resolution investigation.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Experiment_3_Uncertainty_best.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "feature_representation_investigation")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    adapter.load_state_dict(state["model_state_dict"], strict=False)
    adapter.eval()
    backbone = adapter.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    train_loader = DataLoader(SFTL54PairDataset(split="train"), args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SFTL54PairDataset(split="val"), args.batch_size, shuffle=False, num_workers=0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for resolution, (channels, _) in RESOLUTIONS.items():
        probe = None
        if resolution != "7x7":
            (args.output_root / resolution).mkdir(parents=True, exist_ok=True)
            probe = ResolutionRegionalProbe(channels, resolution).to(device)
            history = train_probe(probe, backbone, train_loader, device, args.epochs, args.learning_rate)
            pd.DataFrame(history).to_csv(args.output_root / resolution / "training_history.csv", index=False)
            torch.save(
                {"model_state_dict": probe.state_dict(), "resolution": resolution, "seed": args.seed},
                args.output_root / resolution / "probe_checkpoint.pt",
            )
        _, summary = analyze_resolution(
            resolution, adapter, backbone, probe, val_loader, device, args.output_root
        )
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_root / "comparison_summary.csv", index=False)
    lines = [
        "# E3-C feature representation investigation",
        "",
        "The 7x7 result uses the preserved best E3 checkpoint. The 14x14 and "
        "28x28 probes use frozen EdgeFace backbone maps, a new 1x1 projection "
        "to 100 dimensions, and the unchanged E3 uncertainty matching objective. "
        "Their probe checkpoints are separate and do not overwrite existing artifacts.",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
        "Interpretation must be based on the per-resolution CSV files. This "
        "investigation does not implement reliability-weighted alignment.",
    ]
    (args.output_root / "feature_representation_investigation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
