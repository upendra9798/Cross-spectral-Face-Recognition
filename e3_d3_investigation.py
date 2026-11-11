from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from e3_d2_train import (
    CHECKPOINT,
    FEATURE_ROOT,
    HeadPair,
    THRESHOLDS,
    load_initial_heads,
)
from e3_d_objectives import (
    combined_normalized_ranking,
    normalized_heteroscedastic_fixed,
    pairwise_ranking,
)
from regional_feature_extractor import REGION_NAMES


ROOT = Path(__file__).parent
E3_D2_ROOT = ROOT / "e3_d_objective_diagnostic" / "e3_d2"
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_d3"
VARIANTS = ("ranking_p50", "combined_p50")
REGION_NAMES = tuple(REGION_NAMES)


def corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def rank_values(x: torch.Tensor) -> torch.Tensor:
    return x.argsort().argsort().float()


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return corr(rank_values(x.flatten()), rank_values(y.flatten()))


def load_split(split: str) -> dict[str, object]:
    return torch.load(
        FEATURE_ROOT / f"{split}_features.pt",
        map_location="cpu",
        weights_only=False,
    )


def load_heads(variant: str) -> HeadPair:
    payload = torch.load(
        E3_D2_ROOT / "e3_d2_checkpoints" / f"{variant}_heads.pt",
        map_location="cpu",
        weights_only=False,
    )
    heads = HeadPair()
    heads.load_state_dict(payload["head_state_dict"])
    heads.eval()
    return heads


def uncertainty_outputs(
    heads: HeadPair, artifact: dict[str, object]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        rgb, thermal, combined = heads(
            artifact["rgb_regional_features"].float(),
            artifact["thermal_regional_features"].float(),
        )
    return rgb, thermal, combined


def region_ranking_accuracy(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    threshold: float,
    region_index: int,
) -> tuple[float, int, int]:
    correct = 0
    valid = 0
    for other in range(4):
        if other == region_index:
            continue
        delta_distance = distance[:, region_index] - distance[:, other]
        mask = delta_distance.abs() > threshold
        delta_uncertainty = uncertainty[mask, region_index] - uncertainty[mask, other]
        correct += int((delta_uncertainty * delta_distance[mask] > 0).sum())
        valid += int(mask.sum())
    return (correct / valid if valid else float("nan"), correct, valid)


def region_rows(variant: str) -> list[dict[str, object]]:
    heads = load_heads(variant)
    rows = []
    for split in ("train", "val", "test"):
        artifact = load_split(split)
        rgb, thermal, combined = uncertainty_outputs(heads, artifact)
        distance = artifact["matching_distances"].float()
        subject_ids = artifact["subject_id"]
        for index, region in enumerate(REGION_NAMES):
            ranking, correct, valid = region_ranking_accuracy(
                combined, distance, THRESHOLDS["p50"], index
            )
            rows.append({
                "variant": variant,
                "split": split,
                "region": region,
                "sample_count": int(distance.shape[0]),
                "mean_distance": float(distance[:, index].mean()),
                "distance_std": float(distance[:, index].std(unbiased=False)),
                "mean_rgb_uncertainty": float(rgb[:, index].mean()),
                "rgb_uncertainty_std": float(rgb[:, index].std(unbiased=False)),
                "mean_thermal_uncertainty": float(thermal[:, index].mean()),
                "thermal_uncertainty_std": float(thermal[:, index].std(unbiased=False)),
                "mean_combined_uncertainty": float(combined[:, index].mean()),
                "combined_uncertainty_std": float(combined[:, index].std(unbiased=False)),
                "rgb_pearson": corr(rgb[:, index], distance[:, index]),
                "thermal_pearson": corr(thermal[:, index], distance[:, index]),
                "combined_pearson": corr(combined[:, index], distance[:, index]),
                "rgb_spearman": spearman(rgb[:, index], distance[:, index]),
                "thermal_spearman": spearman(thermal[:, index], distance[:, index]),
                "combined_spearman": spearman(combined[:, index], distance[:, index]),
                "ranking_accuracy": ranking,
                "valid_ranking_pairs": valid,
                "ranking_correct_pairs": correct,
                "subject_count": int(torch.unique(subject_ids).numel()),
            })
    return rows


def subject_consistency_rows(variant: str) -> list[dict[str, object]]:
    heads = load_heads(variant)
    rows = []
    for split in ("train", "val", "test"):
        artifact = load_split(split)
        rgb, thermal, combined = uncertainty_outputs(heads, artifact)
        distance = artifact["matching_distances"].float()
        subjects = artifact["subject_id"]
        for subject in torch.unique(subjects).tolist():
            mask = subjects == subject
            rows.append({
                "variant": variant,
                "split": split,
                "subject_id": int(subject),
                "sample_count": int(mask.sum()),
                "mean_combined_uncertainty": float(combined[mask].mean()),
                "std_combined_uncertainty": float(combined[mask].std(unbiased=False)),
                "mean_matching_distance": float(distance[mask].mean()),
                "subject_mean_error": float(
                    abs(combined[mask].mean() - distance[mask].mean())
                ),
            })
    return rows


def gradient_rows() -> list[dict[str, object]]:
    base, _ = load_initial_heads()
    artifact = load_split("train")
    rgb_features = artifact["rgb_regional_features"][:256].float()
    thermal_features = artifact["thermal_regional_features"][:256].float()
    distance = artifact["matching_distances"][:256].float()
    rgb_u, thermal_u, combined = base(rgb_features, thermal_features)
    rgb_u.retain_grad()
    thermal_u.retain_grad()
    combined.retain_grad()
    normalized, _ = normalized_heteroscedastic_fixed(
        combined, distance, 0.00135366382334
    )
    ranking_details = pairwise_ranking(combined, distance, THRESHOLDS["p50"])
    combined_loss, _, _ = combined_normalized_ranking(
        combined, distance, 0.00135366382334, THRESHOLDS["p50"]
    )
    losses = {
        "normalized_fixed": normalized,
        "ranking_p50": ranking_details.loss,
        "combined_p50": combined_loss,
    }
    gradients = {}
    rows = []
    for name, loss in losses.items():
        parameter_gradients = torch.autograd.grad(
            loss, tuple(base.parameters()), retain_graph=True, allow_unused=True
        )
        output_gradient = torch.autograd.grad(
            loss, combined, retain_graph=True
        )[0]
        gradient_norm = torch.sqrt(
            sum(
                gradient.pow(2).sum()
                for gradient in parameter_gradients
                if gradient is not None
            )
        )
        gradients[name] = output_gradient.detach()
        for region_index, region in enumerate(REGION_NAMES):
            rows.append({
                "component": name,
                "region": region,
                "loss": float(loss.detach()),
                "parameter_gradient_norm": float(gradient_norm),
                "output_gradient_mean": float(output_gradient[:, region_index].mean()),
                "output_gradient_std": float(output_gradient[:, region_index].std(unbiased=False)),
                "output_gradient_abs_mean": float(output_gradient[:, region_index].abs().mean()),
                "valid_ranking_pairs": ranking_details.valid_pairs if "ranking" in name else None,
            })
    for left, right in (
        ("normalized_fixed", "ranking_p50"),
        ("normalized_fixed", "combined_p50"),
        ("ranking_p50", "combined_p50"),
    ):
        left_gradient = gradients[left].flatten()
        right_gradient = gradients[right].flatten()
        cosine = F.cosine_similarity(left_gradient[None], right_gradient[None]).item()
        rows.append({
            "component": f"cosine:{left}:{right}",
            "region": "all",
            "loss": float("nan"),
            "parameter_gradient_norm": float("nan"),
            "output_gradient_mean": float("nan"),
            "output_gradient_std": float("nan"),
            "output_gradient_abs_mean": cosine,
            "valid_ranking_pairs": None,
        })
    return rows


def corruption_rows() -> list[dict[str, object]]:
    source = pd.read_csv(E3_D2_ROOT / "e3_d2_corruption_results.csv")
    source = source[source["variant"].isin(VARIANTS)].copy()
    source["distance_change"] = source["target_region_distance_change"]
    source["corrupted_region"] = source["region"]
    source["is_positive_combined_change"] = source["combined_uncertainty_change"] > 0
    source["uncertainty_distance_sign_agreement"] = (
        source["combined_uncertainty_change"] * source["distance_change"] > 0
    )
    return source[
        [
            "variant",
            "pair_key",
            "corrupted_region",
            "sigma",
            "distance_change",
            "rgb_uncertainty_change",
            "thermal_uncertainty_change",
            "combined_uncertainty_change",
            "is_positive_combined_change",
            "uncertainty_distance_sign_agreement",
        ]
    ].to_dict("records")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    start = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_region_rows = []
    all_subject_rows = []
    for variant in VARIANTS:
        all_region_rows.extend(region_rows(variant))
        all_subject_rows.extend(subject_consistency_rows(variant))
    write_csv(OUTPUT / "e3_d3_region_analysis.csv", all_region_rows)
    write_csv(OUTPUT / "e3_d3_ranking_generalization.csv", all_subject_rows)
    write_csv(OUTPUT / "e3_d3_corruption_analysis.csv", corruption_rows())
    write_csv(OUTPUT / "e3_d3_gradient_comparison.csv", gradient_rows())

    metric_audit = """# E3-D3 metric audit

## Current E3-D2 definitions

The current evaluator is `e3_d2_evaluate.py`.

### Pearson correlation

`evaluate_clean()` flattens the complete `[N, 4]` uncertainty tensor and the
complete `[N, 4]` matching-distance tensor before calling `torch.corrcoef`.
Therefore the reported Pearson values are global sample-region correlations.
They are not per-region, per-sample, or correlations of aggregated means.
RGB, thermal, and combined uncertainty are each available separately, but the
main D2 result table reports combined uncertainty.

### Spearman correlation

Spearman is computed by applying `argsort().argsort()` to the flattened
uncertainty and distance tensors and then calling the same Pearson helper.
It is also a global sample-region statistic. Ties are not explicitly handled
with average ranks; ordinal ranks are assigned by the double argsort.

### Ranking accuracy

`ranking_accuracy()` operates within each sample. It compares all six unordered
region pairs, retains pairs with `abs(distance_i-distance_j) > threshold`, and
counts whether the uncertainty difference has the same sign as the distance
difference. Counts are pooled across samples and region pairs. Exact ties and
pairs below threshold are excluded. The D2 table uses combined uncertainty.

### Reliability standard deviation

Reliability is computed as `softmax(-combined_uncertainty, dim=1)` and the
reported standard deviation is over all samples and all four regions. It is not
a per-region or per-sample average.

### Corruption uncertainty change

The D2 corruption evaluator corrupts one target region at a time and records
the uncertainty and distance change for that target region only. It does not
record unrelated-region responses. The CSV contains RGB, thermal, and combined
changes, with validation samples and sigma 3, 7, and 15.

## Metric comparability

The global correlations, pooled within-sample ranking accuracy, global
reliability standard deviation, and target-region corruption changes are
different aggregation levels. They must not be interpreted as interchangeable
measurements.
"""
    (OUTPUT / "e3_d3_metric_audit.md").write_text(metric_audit, encoding="utf-8")

    recommendation = f"""# E3-D3 training recommendation

No longer training was launched.

## Evidence

The ranking-only p50 checkpoint is the strongest short-run discriminator:
it has higher ranking accuracy and a larger positive corruption response than
the combined p50 checkpoint. However, its global clean correlations remain
near zero, so it has not demonstrated calibration.

The combined p50 normalized component has a gradient cosine comparison against
the ranking component in `e3_d3_gradient_comparison.csv`. These values should
be interpreted as directional diagnostics, not proof of suppression.

## Proposed next experiment

- Train only the uncertainty heads for 10 epochs initially.
- Keep the p50 threshold `{THRESHOLDS["p50"]}` and ranking weight unchanged.
- Keep Adam, learning rate `0.001`, seed `42`, batch size `64`, and shuffle disabled.
- Evaluate after every epoch.
- Select checkpoints by validation combined Spearman correlation, subject to:
  - positive Pearson and Spearman,
  - ranking accuracy above the three-epoch p50 reference `0.756190`,
  - positive uncertainty change for the directly corrupted region at held-out
    blur strengths,
  - no increase in clamp-hit rate,
  - no collapse of reliability variation.
- Use patience 3 early stopping on validation Spearman.
- Do not use test results for selection.

The current three-epoch ranking p50 checkpoint remains the reference. This
recommendation is provisional and does not authorize the longer run or Phase 4.
"""
    (OUTPUT / "e3_d3_training_recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    print(f"E3-D3 completed in {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
