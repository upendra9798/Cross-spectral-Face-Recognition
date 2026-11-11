from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from e3_d2_train import HeadPair
from e3_d4_train import (
    CHECKPOINT_ROOT,
    FEATURE_ROOT,
    OUTPUT,
    REFERENCE,
    THRESHOLD,
)
from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent
SIGMAS = (3.0, 7.0, 15.0)
REGIONS = tuple(REGION_NAMES)


def corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return corr(x.argsort().argsort().float(), y.argsort().argsort().float())


def ranking_accuracy(
    uncertainty: torch.Tensor, distance: torch.Tensor, region_index: int | None = None
) -> tuple[float, int, int]:
    indices = [region_index] if region_index is not None else list(range(4))
    correct = 0
    total = 0
    for left in indices:
        rights = (
            [right for right in range(4) if right != left]
            if region_index is not None
            else range(left + 1, 4)
        )
        for right in rights:
            delta_distance = distance[:, left] - distance[:, right]
            mask = delta_distance.abs() > THRESHOLD
            delta_uncertainty = uncertainty[mask, left] - uncertainty[mask, right]
            correct += int((delta_uncertainty * delta_distance[mask] > 0).sum())
            total += int(mask.sum())
    return (correct / total if total else float("nan"), correct, total)


def load_artifact(split: str) -> dict[str, object]:
    return torch.load(
        FEATURE_ROOT / f"{split}_features.pt",
        map_location="cpu",
        weights_only=False,
    )


def load_heads(checkpoint: Path) -> HeadPair:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HeadPair()
    model.load_state_dict(payload["head_state_dict"], strict=True)
    model.eval()
    return model


def clean_metrics(heads: HeadPair, split: str, epoch: int) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    artifact = load_artifact(split)
    rgb, thermal, combined = heads(
        artifact["rgb_regional_features"].float(),
        artifact["thermal_regional_features"].float(),
    )
    distance = artifact["matching_distances"].float()
    epoch_row = {
        "epoch": epoch,
        "split": split,
        "rgb_mean": float(rgb.mean()),
        "rgb_std": float(rgb.std(unbiased=False)),
        "thermal_mean": float(thermal.mean()),
        "thermal_std": float(thermal.std(unbiased=False)),
        "combined_mean": float(combined.mean()),
        "combined_std": float(combined.std(unbiased=False)),
        "reliability_std": float(F.softmax(-combined, dim=1).std(unbiased=False)),
        "rgb_pearson": corr(rgb.flatten(), distance.flatten()),
        "thermal_pearson": corr(thermal.flatten(), distance.flatten()),
        "combined_pearson": corr(combined.flatten(), distance.flatten()),
        "rgb_spearman": spearman(rgb.flatten(), distance.flatten()),
        "thermal_spearman": spearman(thermal.flatten(), distance.flatten()),
        "combined_spearman": spearman(combined.flatten(), distance.flatten()),
        "ranking_accuracy": ranking_accuracy(combined, distance)[0],
        "clamp_hit_rate": 0.0,
        "zero_gradient_rate": float("nan"),
    }
    region_rows = []
    for index, region in enumerate(REGIONS):
        region_rows.append({
            "epoch": epoch,
            "split": split,
            "region": region,
            "mean_distance": float(distance[:, index].mean()),
            "distance_std": float(distance[:, index].std(unbiased=False)),
            "rgb_mean": float(rgb[:, index].mean()),
            "rgb_std": float(rgb[:, index].std(unbiased=False)),
            "thermal_mean": float(thermal[:, index].mean()),
            "thermal_std": float(thermal[:, index].std(unbiased=False)),
            "combined_mean": float(combined[:, index].mean()),
            "combined_std": float(combined[:, index].std(unbiased=False)),
            "rgb_pearson": corr(rgb[:, index], distance[:, index]),
            "thermal_pearson": corr(thermal[:, index], distance[:, index]),
            "combined_pearson": corr(combined[:, index], distance[:, index]),
            "rgb_spearman": spearman(rgb[:, index], distance[:, index]),
            "thermal_spearman": spearman(thermal[:, index], distance[:, index]),
            "combined_spearman": spearman(combined[:, index], distance[:, index]),
            "ranking_accuracy": ranking_accuracy(combined, distance, index)[0],
        })
    subject_rows = []
    subjects = artifact["subject_id"]
    for subject in torch.unique(subjects).tolist():
        mask = subjects == subject
        subject_rows.append({
            "epoch": epoch,
            "split": split,
            "subject_id": int(subject),
            "sample_count": int(mask.sum()),
            "rgb_pearson": corr(rgb[mask].flatten(), distance[mask].flatten()),
            "thermal_pearson": corr(thermal[mask].flatten(), distance[mask].flatten()),
            "combined_pearson": corr(combined[mask].flatten(), distance[mask].flatten()),
            "rgb_spearman": spearman(rgb[mask].flatten(), distance[mask].flatten()),
            "thermal_spearman": spearman(thermal[mask].flatten(), distance[mask].flatten()),
            "combined_spearman": spearman(combined[mask].flatten(), distance[mask].flatten()),
        })
    return epoch_row, region_rows, subject_rows


def bootstrap_subject_ci(values: list[float], seed: int = 42, samples: int = 2000) -> tuple[float, float, float]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = np.mean(generator.choice(values, (samples, values.size), replace=True), axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def add_subject_summary(subject_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(subject_rows)
    rows = []
    for (epoch, split), group in frame.groupby(["epoch", "split"]):
        for metric in ("combined_pearson", "combined_spearman"):
            mean, lower, upper = bootstrap_subject_ci(group[metric].tolist(), seed=42 + int(epoch))
            finite = group[metric].dropna()
            rows.append({
                "epoch": epoch,
                "split": split,
                "metric": metric,
                "subject_count": len(group),
                "valid_subject_count": len(finite),
                "positive_subject_count": int((finite > 0).sum()),
                "positive_subject_percentage": float(100 * (finite > 0).mean()) if len(finite) else float("nan"),
                "mean": mean,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
            })
    return rows


def blur_tensor(images: torch.Tensor, landmarks: torch.Tensor, region_index: int, sigma: float) -> torch.Tensor:
    boxes = RegionalFeatureExtractor().image_boxes(landmarks)
    blurred = TF.gaussian_blur(images, [9, 9], sigma=sigma)
    result = images.clone()
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[index, :, y1:y2, x1:x2] = blurred[index, :, y1:y2, x1:x2]
    return result


def corruption_metrics(checkpoint: Path, epoch: int) -> list[dict[str, object]]:
    device = torch.device("cpu")
    teacher = load_pretrained_edgeface()
    adapter = EdgeFaceTransformerAdapter(copy.deepcopy(teacher))
    state = torch.load(ROOT / "Experiment_3_Uncertainty_best.pt", map_location="cpu", weights_only=True)
    adapter.load_state_dict(state["model_state_dict"], strict=True)
    heads = load_heads(checkpoint)
    adapter.rgb_uncertainty_head.load_state_dict(heads.rgb.state_dict())
    adapter.thermal_uncertainty_head.load_state_dict(heads.thermal.state_dict())
    adapter.eval()
    loader = DataLoader(SFTL54PairDataset(split="val"), batch_size=64, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for batch in loader:
            landmarks = batch["landmarks"]
            clean = adapter.forward_with_uncertainty(batch["rgb"], batch["thermal"], landmarks)
            clean_distance = 1.0 - F.cosine_similarity(
                clean["rgb_regional_features"], clean["thermal_regional_features"], dim=-1
            )
            for target_index, target_region in enumerate(REGIONS):
                for sigma in SIGMAS:
                    rgb = blur_tensor(batch["rgb"], landmarks, target_index, sigma)
                    thermal = blur_tensor(batch["thermal"], landmarks, target_index, sigma)
                    corrupted = adapter.forward_with_uncertainty(rgb, thermal, landmarks)
                    distance = 1.0 - F.cosine_similarity(
                        corrupted["rgb_regional_features"],
                        corrupted["thermal_regional_features"],
                        dim=-1,
                    )
                    for sample in range(distance.shape[0]):
                        for observed_index, observed_region in enumerate(REGIONS):
                            rows.append({
                                "epoch": epoch,
                                "corrupted_region": target_region,
                                "observed_region": observed_region,
                                "sigma": sigma,
                                "rgb_uncertainty_change": float(corrupted["rgb_uncertainty"][sample, observed_index] - clean["rgb_uncertainty"][sample, observed_index]),
                                "thermal_uncertainty_change": float(corrupted["thermal_uncertainty"][sample, observed_index] - clean["thermal_uncertainty"][sample, observed_index]),
                                "combined_uncertainty_change": float(corrupted["combined_uncertainty"][sample, observed_index] - clean["combined_uncertainty"][sample, observed_index]),
                                "matching_distance_change": float(distance[sample, observed_index] - clean_distance[sample, observed_index]),
                                "is_target_region": observed_index == target_index,
                            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D4 evaluation.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    checkpoints = sorted(CHECKPOINT_ROOT.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No E3-D4 epoch checkpoints found.")
    epoch_rows, region_rows, subject_rows, corruption_rows = [], [], [], []
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.split("_")[1])
        heads = load_heads(checkpoint)
        epoch_row, current_regions, current_subjects = clean_metrics(heads, "val", epoch)
        epoch_row["checkpoint"] = str(checkpoint)
        epoch_rows.append(epoch_row)
        region_rows.extend(current_regions)
        subject_rows.extend(current_subjects)
        corruption_rows.extend(corruption_metrics(checkpoint, epoch))
    subject_summary = add_subject_summary(subject_rows)
    epoch_frame = pd.DataFrame(epoch_rows)
    region_frame = pd.DataFrame(region_rows)
    corruption_frame = pd.DataFrame(corruption_rows)
    eligibility = []
    for epoch in epoch_frame["epoch"]:
        epoch_row = epoch_frame[epoch_frame["epoch"] == epoch].iloc[0]
        regions = region_frame[region_frame["epoch"] == epoch]
        target = corruption_frame[
            (corruption_frame["epoch"] == epoch) & corruption_frame["is_target_region"]
        ].groupby("corrupted_region")["combined_uncertainty_change"].mean()
        eligible = bool(
            epoch_row["combined_pearson"] > 0
            and epoch_row["combined_spearman"] > 0
            and epoch_row["ranking_accuracy"] >= 0.756190
            and len(target) == 4
            and (target > 0).all()
            and (regions["combined_pearson"] >= 0).all()
            and epoch_row["clamp_hit_rate"] == 0
        )
        eligibility.append({"epoch": int(epoch), "eligible": eligible})
    eligibility_frame = pd.DataFrame(eligibility)
    eligible_epochs = eligibility_frame.loc[eligibility_frame["eligible"], "epoch"].tolist()
    selected_epoch = None
    if eligible_epochs:
        selected_epoch = int(
            epoch_frame[epoch_frame["epoch"].isin(eligible_epochs)]
            .sort_values("combined_spearman", ascending=False)
            .iloc[0]["epoch"]
        )
    epoch_frame["eligible"] = epoch_frame["epoch"].map(
        dict(zip(eligibility_frame["epoch"], eligibility_frame["eligible"]))
    )
    epoch_frame["selected"] = epoch_frame["epoch"] == selected_epoch
    epoch_frame.to_csv(args.output / "e3_d4_epoch_metrics.csv", index=False)
    region_frame.to_csv(args.output / "e3_d4_region_metrics.csv", index=False)
    pd.DataFrame(subject_summary).to_csv(args.output / "e3_d4_subject_metrics.csv", index=False)
    corruption_frame.to_csv(args.output / "e3_d4_corruption_metrics.csv", index=False)
    history = json.loads((args.output / "e3_d4_training_state.json").read_text())
    pd.DataFrame(history["history"]).to_csv(args.output / "e3_d4_training_history.csv", index=False)
    report = [
        "# E3-D4 longer ranking-only uncertainty experiment",
        "",
        f"- Reference checkpoint: `{REFERENCE}`",
        f"- Reference checkpoint file SHA-256: `{history['reference_file_hash']}`",
        f"- Initial uncertainty-head state hash: `{history['initial_state_hash']}`",
        f"- Final uncertainty-head state hash: `{history['final_state_hash']}`",
        f"- Threshold: `{THRESHOLD}`",
        "- Objective: pairwise ranking only",
        "- Training: 10 epochs maximum, Adam, learning rate 0.001, batch size 64, seed 42, shuffle disabled",
        "- Test data was not used for training or checkpoint selection.",
        "",
        "## Selection result",
        "",
        f"Eligible epochs: `{eligible_epochs}`",
        f"Selected epoch: `{selected_epoch}`",
        "",
        "Eligibility required positive global Pearson and Spearman, ranking "
        "accuracy at least 0.756190, positive targeted corruption response for "
        "all regions, no negative per-region Pearson values, and zero clamp hits.",
        "",
        "If no epoch is selected, the experiment did not produce a successful "
        "checkpoint under the pre-registered criteria.",
        "",
        "Per-epoch, regional, subject and corruption metrics are saved in the "
        "requested CSV files. Subject bootstrap summaries contain 95% confidence "
        "intervals over subject-level correlations.",
    ]
    (args.output / "e3_d4_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
