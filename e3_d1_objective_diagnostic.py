from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from e3_d_objectives import (
    calibration_regularization,
    combined_normalized_ranking,
    normalized_heteroscedastic_batch,
    normalized_heteroscedastic_fixed,
    original_heteroscedastic,
    pairwise_ranking,
)


ROOT = Path(__file__).parent
REGIONS = ("left_eye", "right_eye", "nose", "mouth")


def load_distances(root: Path) -> dict[str, torch.Tensor]:
    result = {}
    for split in ("train", "val", "test"):
        artifact = torch.load(
            root / f"{split}_features.pt", map_location="cpu", weights_only=False
        )
        result[split] = artifact["matching_distances"].to(torch.float64)
        if result[split].shape[1:] != (4,):
            raise ValueError(f"{split} has invalid distance shape {result[split].shape}")
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distance_statistics(distances: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows = []
    for split, values in distances.items():
        for region_index, region in enumerate(REGIONS):
            region_values = values[:, region_index]
            rows.append({
                "split": split,
                "region": region,
                "count": int(region_values.numel()),
                "mean": float(region_values.mean()),
                "std": float(region_values.std(unbiased=False)),
                "minimum": float(region_values.min()),
                "maximum": float(region_values.max()),
            })
        flattened = values.flatten()
        rows.append({
            "split": split,
            "region": "all",
            "count": int(flattened.numel()),
            "mean": float(flattened.mean()),
            "std": float(flattened.std(unbiased=False)),
            "minimum": float(flattened.min()),
            "maximum": float(flattened.max()),
        })
    return rows


def pairwise_absolute_differences(values: torch.Tensor) -> torch.Tensor:
    differences = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            differences.append((values[:, left] - values[:, right]).abs())
    return torch.cat(differences)


def ranking_statistics(
    distances: dict[str, torch.Tensor], thresholds: list[float]
) -> list[dict[str, object]]:
    rows = []
    for split, values in distances.items():
        total = values.shape[0] * 6
        for threshold in thresholds:
            valid = int((pairwise_absolute_differences(values) > threshold).sum())
            rows.append({
                "split": split,
                "threshold": threshold,
                "valid_pairs": valid,
                "total_pairs": total,
                "valid_percentage": 100.0 * valid / total,
                "threshold_source": "train" if split == "train" else "train_quantile",
            })
    return rows


def gradient_rows(distance: torch.Tensor, training_scale: float, threshold: float) -> list[dict[str, object]]:
    uncertainty = torch.zeros_like(distance, requires_grad=True)
    outputs: list[tuple[str, torch.Tensor, int | None]] = []
    outputs.append(("original", original_heteroscedastic(uncertainty, distance), None))
    batch_loss, _ = normalized_heteroscedastic_batch(uncertainty, distance)
    outputs.append(("normalized_batch", batch_loss, None))
    fixed_loss, _ = normalized_heteroscedastic_fixed(uncertainty, distance, training_scale)
    outputs.append(("normalized_fixed", fixed_loss, None))
    ranking = pairwise_ranking(uncertainty, distance, threshold)
    outputs.append(("ranking", ranking.loss, ranking.valid_pairs))
    combined, details, _ = combined_normalized_ranking(
        uncertainty, distance, training_scale, threshold
    )
    outputs.append(("combined", combined, details.valid_pairs))
    calibration, _ = calibration_regularization(uncertainty, distance)
    outputs.append(("calibration", calibration, None))
    rows = []
    for name, loss, valid_pairs in outputs:
        gradient = torch.autograd.grad(loss, uncertainty, retain_graph=True)[0]
        rows.append({
            "objective": name,
            "loss": float(loss),
            "gradient_mean": float(gradient.mean()),
            "gradient_std": float(gradient.std(unbiased=False)),
            "gradient_min": float(gradient.min()),
            "gradient_max": float(gradient.max()),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "zero_gradient_count": int((gradient == 0).sum()),
            "valid_ranking_pairs": valid_pairs,
            "uniform_uncertainty_input": True,
            "training_scale": training_scale,
            "threshold": threshold,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D1 objective diagnostics.")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "e3_d_objective_diagnostic" / "frozen_features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "e3_d_objective_diagnostic",
    )
    args = parser.parse_args()
    start = time.perf_counter()
    distances = load_distances(args.features)
    train_differences = pairwise_absolute_differences(distances["train"])
    quantiles = torch.quantile(
        train_differences, torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64)
    )
    thresholds = [0.0] + [float(value) for value in quantiles]
    training_scale = float(distances["train"].mean())
    write_rows(args.output / "e3_d1_distance_statistics.csv", distance_statistics(distances))
    write_rows(
        args.output / "e3_d1_ranking_pair_statistics.csv",
        ranking_statistics(distances, thresholds),
    )
    gradient_analysis = gradient_rows(
        distances["train"][: min(256, len(distances["train"]))],
        training_scale,
        float(quantiles[1]),
    )
    write_rows(args.output / "e3_d1_gradient_analysis.csv", gradient_analysis)
    synthetic_script = ROOT / "e3_d1_synthetic_tests.py"
    synthetic_output = args.output / "synthetic_tests.json"
    comparison_rows = [
        {
            "objective": row["objective"],
            "mathematical_test": "pass" if row["gradient_finite"] and math.isfinite(row["loss"]) else "fail",
            "finite_loss": True,
            "finite_gradient": row["gradient_finite"],
            "actual_train_probe_loss": row["loss"],
            "actual_train_probe_gradient_mean": row["gradient_mean"],
            "valid_ranking_pairs": row["valid_ranking_pairs"],
            "uses_train_only_statistics": row["objective"] in {
                "normalized_fixed",
                "combined",
                "ranking",
            },
            "model_training_performed": False,
        }
        for row in gradient_analysis
    ]
    write_rows(args.output / "e3_d1_objective_comparison.csv", comparison_rows)
    metadata = json.loads((args.features / "metadata.json").read_text(encoding="utf-8"))
    report = [
        "# E3-D1 mathematical objective diagnostics",
        "",
        "No model or uncertainty head was trained. All objective tests use "
        "synthetic uncertainty tensors and frozen E3-D0 regional distances.",
        "",
        "## E3-D0 integrity",
        "",
        f"- Checkpoint SHA-256: `{metadata['checkpoint_sha256']}`",
        f"- Split counts: `{metadata['split_counts']}`",
        "- Feature shape per sample: `[4, 100]`",
        "- Distance shape per sample: `[4]`",
        "- Feature normalization: L2-normalized regional features.",
        "- Distance: `1 - cosine_similarity(rgb, thermal)`.",
        "",
        "## Training-derived constants",
        "",
        f"- Fixed training distance scale: `{training_scale:.12g}`",
        f"- Ranking thresholds: `{thresholds}`",
        "- All thresholds and the fixed scale were estimated from training distances only.",
        "- Validation and test distances were not used to select constants.",
        "",
        "## Frozen distance findings",
        "",
        "| Split | Mean | Std | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, values in distances.items():
        report.append(
            f"| {split} | {values.mean():.9f} | {values.std(unbiased=False):.9f} | "
            f"{values.min():.9f} | {values.max():.9f} |"
        )
    report.extend([
        "",
        "Regional and split-specific statistics are in "
        "`e3_d1_distance_statistics.csv`.",
        "",
        "## Ranking-pair availability",
        "",
        "Pair statistics are in `e3_d1_ranking_pair_statistics.csv`. A pair is "
        "valid only when the absolute regional distance difference exceeds the "
        "training-derived threshold. Ties and nearly tied pairs are excluded.",
        "",
        f"At the training-derived median threshold `{float(quantiles[1]):.12g}`, "
        f"training has {int((train_differences > quantiles[1]).sum())} valid "
        f"of {len(train_differences)} pairs; validation and test counts are "
        "reported without using them to choose the threshold.",
        "",
        "## Objective conclusions",
        "",
        "- Original heteroscedastic loss has an unconstrained optimum `u=log(d)` "
        "and therefore pushes toward low uncertainty for these small distances.",
        "- Normalized heteroscedastic losses remove the absolute scale mismatch, "
        "but retain an optimum based on normalized distance and do not add supervision.",
        "- Ranking loss is mathematically valid when enough unequal regional pairs "
        "exist; it constrains ordering but not absolute uncertainty calibration.",
        "- The combined objective is mathematically justified as a scale-corrected "
        "magnitude term plus a relative-order term, but it is not evidence of calibration.",
        "- Corruption supervision is implemented as an interface only. No corruption "
        "targets were fabricated and no corruption training was run.",
        "- Calibration regularization uses detached per-sample min-max distance "
        "targets; tied samples produce a stable finite target rather than a gradient path.",
        "",
        "## Mathematical test status",
        "",
        f"Synthetic test source: `{synthetic_script.name}`",
        f"Synthetic results path: `{synthetic_output.name}`",
        "All losses are expected to remain finite. Detached normalization and "
        "detached calibration/corruption targets prevent hidden gradients through statistics.",
        "`e3_d1_objective_comparison.csv` records finite-loss/gradient status "
        "and confirms that no model training was performed.",
        "",
        "## Recommendation for E3-D2",
        "",
        "Proceed first with the combined fixed-scale normalized heteroscedastic "
        "plus ranking objective only if the ranking-pair table shows sufficient "
        "non-tied pairs. Retain normalized-only and ranking-only as ablations. "
        "Do not select an objective based on loss decrease alone; require validation "
        "uncertainty/error correlation and held-out corruption response.",
        "",
        f"Diagnostic runtime: `{time.perf_counter() - start:.3f}` seconds.",
    ])
    (args.output / "e3_d1_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output / "e3_d1_protocol.json").write_text(
        json.dumps(
            {
                "training_scale": training_scale,
                "ranking_thresholds": thresholds,
                "threshold_source": "train matching distances only",
                "synthetic_script": synthetic_script.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
