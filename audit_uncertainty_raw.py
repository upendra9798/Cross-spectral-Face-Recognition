from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent
STRENGTHS = (3.0, 7.0, 15.0)
MIN_LOG_U = -4.0
MAX_LOG_U = 4.0


def load_model(checkpoint: Path, device: torch.device) -> EdgeFaceTransformerAdapter:
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=False)
    return model.eval()


def blur_region(
    images: torch.Tensor,
    landmarks: torch.Tensor,
    region_index: int,
    sigma: float,
) -> torch.Tensor:
    extractor = RegionalFeatureExtractor()
    boxes = extractor.image_boxes(landmarks)
    blurred = TF.gaussian_blur(images, kernel_size=[9, 9], sigma=sigma)
    result = images.clone()
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[index, :, y1:y2, x1:x2] = blurred[index, :, y1:y2, x1:x2]
    return result


def capture(
    model: EdgeFaceTransformerAdapter,
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    landmarks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model.forward_with_uncertainty(rgb, thermal, landmarks)
        raw_rgb = model.rgb_uncertainty_head.network(output["rgb_regional_features"]).squeeze(-1)
        raw_thermal = model.thermal_uncertainty_head.network(
            output["thermal_regional_features"]
        ).squeeze(-1)
        clamped_rgb = raw_rgb.clamp(MIN_LOG_U, MAX_LOG_U)
        clamped_thermal = raw_thermal.clamp(MIN_LOG_U, MAX_LOG_U)
        combined = clamped_rgb + clamped_thermal
        reliability = torch.softmax(-combined, dim=1)
        distance = 1.0 - F.cosine_similarity(
            output["rgb_regional_features"],
            output["thermal_regional_features"],
            dim=-1,
        )
    return {
        "raw_rgb": raw_rgb,
        "raw_thermal": raw_thermal,
        "clamped_rgb": clamped_rgb,
        "clamped_thermal": clamped_thermal,
        "combined": combined,
        "reliability": reliability,
        "distance": distance,
    }


def append_rows(
    rows: list[dict[str, object]],
    batch: dict[str, object],
    values: dict[str, torch.Tensor],
    condition: str,
    sigma: float,
) -> None:
    for sample_index, pair_key in enumerate(batch["pair_key"]):
        for region_index, region in enumerate(REGION_NAMES):
            rows.append(
                {
                    "pair_key": pair_key,
                    "subject_id": int(batch["subject_id"][sample_index]),
                    "condition": condition,
                    "sigma": sigma,
                    "region": region,
                    "raw_rgb": float(values["raw_rgb"][sample_index, region_index]),
                    "raw_thermal": float(values["raw_thermal"][sample_index, region_index]),
                    "clamped_rgb": float(values["clamped_rgb"][sample_index, region_index]),
                    "clamped_thermal": float(values["clamped_thermal"][sample_index, region_index]),
                    "combined_uncertainty": float(values["combined"][sample_index, region_index]),
                    "reliability": float(values["reliability"][sample_index, region_index]),
                    "matching_distance": float(values["distance"][sample_index, region_index]),
                }
            )


def stats_table(report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality in ("rgb", "thermal"):
        column = f"raw_{modality}"
        values = report[column]
        quantiles = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append(
            {
                "modality": modality,
                "mean": values.mean(),
                "std": values.std(),
                "minimum": values.min(),
                "maximum": values.max(),
                "p05": quantiles.loc[0.05],
                "p25": quantiles.loc[0.25],
                "p50": quantiles.loc[0.50],
                "p75": quantiles.loc[0.75],
                "p95": quantiles.loc[0.95],
            }
        )
    return pd.DataFrame(rows)


def region_table(report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, condition_group in report.groupby("condition", sort=False):
        for region, group in condition_group.groupby("region", sort=False):
            rows.append(
                {
                    "condition": condition,
                    "region": region,
                    "mean_raw_rgb": group["raw_rgb"].mean(),
                    "std_raw_rgb": group["raw_rgb"].std(),
                    "mean_raw_thermal": group["raw_thermal"].mean(),
                    "std_raw_thermal": group["raw_thermal"].std(),
                    "mean_matching_distance": group["matching_distance"].mean(),
                    "corr_rgb_distance": group["raw_rgb"].corr(
                        group["matching_distance"]
                    ),
                    "corr_thermal_distance": group["raw_thermal"].corr(
                        group["matching_distance"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def gradient_table(report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_rgb = torch.tensor(report["raw_rgb"].to_numpy(), requires_grad=True)
    raw_thermal = torch.tensor(report["raw_thermal"].to_numpy(), requires_grad=True)
    distance = torch.tensor(report["matching_distance"].to_numpy())
    combined = raw_rgb.clamp(MIN_LOG_U, MAX_LOG_U) + raw_thermal.clamp(
        MIN_LOG_U, MAX_LOG_U
    )
    per_element_loss = torch.exp(-combined) * distance + combined
    loss = per_element_loss.mean()
    rgb_gradient, thermal_gradient = torch.autograd.grad(
        loss, (raw_rgb, raw_thermal)
    )
    for modality, raw, gradient in (
        ("rgb", raw_rgb, rgb_gradient),
        ("thermal", raw_thermal, thermal_gradient),
    ):
        raw_values = raw.detach().numpy()
        # Undo the mean reduction so this is the gradient of one objective term.
        gradients = gradient.detach().numpy() * len(raw_values)
        rows.append(
            {
                "modality": modality,
                "loss": float(loss),
                "objective": "combined_rgb_thermal",
                "gradient_mean": float(gradients.mean()),
                "gradient_std": float(gradients.std()),
                "gradient_min": float(gradients.min()),
                "gradient_max": float(gradients.max()),
                "zero_gradient_count": int((abs(gradients) < 1e-12).sum()),
                "near_lower_clamp_count": int((raw_values <= MIN_LOG_U + 1e-3).sum()),
                "near_upper_clamp_count": int((raw_values >= MAX_LOG_U - 1e-3).sum()),
                "lower_clamped_count": int((raw_values < MIN_LOG_U).sum()),
                "upper_clamped_count": int((raw_values > MAX_LOG_U).sum()),
                "total_count": len(raw_values),
            }
        )
    return pd.DataFrame(rows)


def write_visualizations(report: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in REGION_NAMES:
        group = report[report["region"] == region]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for condition, condition_group in group.groupby("condition", sort=False):
            means = condition_group.groupby("sigma")[["raw_rgb", "raw_thermal"]].mean()
            axes[0].plot(means.index, means["raw_rgb"], marker="o", label=condition)
            axes[1].plot(means.index, means["raw_thermal"], marker="o", label=condition)
        axes[0].set_title(f"{region}: raw RGB uncertainty")
        axes[1].set_title(f"{region}: raw thermal uncertainty")
        for axis in axes:
            axis.set_xlabel("blur sigma")
            axis.set_ylabel("log uncertainty")
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / f"{region}.png", dpi=150)
        plt.close(figure)


def write_report(
    report: pd.DataFrame,
    statistics: pd.DataFrame,
    regions: pd.DataFrame,
    gradients: pd.DataFrame,
    output: Path,
) -> None:
    clean = report[report["condition"] == "clean"]
    raw_std = float(pd.concat([clean["raw_rgb"], clean["raw_thermal"]]).std())
    reliability_std = float(clean["reliability"].std())
    clamp_hit_rate = float(
        (
            (clean[["raw_rgb", "raw_thermal"]] <= MIN_LOG_U)
            | (clean[["raw_rgb", "raw_thermal"]] >= MAX_LOG_U)
        ).to_numpy().mean()
    )
    target = regions[
        regions["condition"].isin(["left_eye", "right_eye", "nose", "mouth"])
    ]
    target_delta = target["mean_matching_distance"].mean() - clean["matching_distance"].mean()
    if reliability_std < 0.01 and raw_std < 0.1:
        conclusion = "A and D: objective formulation is primary, with the resulting collapse amplified by the feature representation."
    elif clamp_hit_rate > 0.25:
        conclusion = "B: clamp saturation is a primary contributor."
    else:
        conclusion = "D: the limitation is most consistent with a combination of objective and feature representation."
    text = f"""# Raw uncertainty and gradient saturation audit

Checkpoint: `Experiment_3_Uncertainty_best.pt`

## Conclusion

**{conclusion}**

Clean raw uncertainty pooled standard deviation was `{raw_std:.6f}` and clean
reliability standard deviation was `{reliability_std:.6f}`. The clean clamp-hit
rate was `{clamp_hit_rate:.6f}`. Mean targeted-corruption matching-distance
change was `{target_delta:.9f}`.

The raw outputs, clamped outputs, combined uncertainty, reliability, and
matching distance were captured on the same validation pairs. Raw values are
the direct final MLP outputs before the head's clamp; clamped values use
`[-4, 4]`.

## Raw statistics

```text
{statistics.to_string(index=False)}
```

## Gradient audit

The gradient audit differentiates the implemented clamped objective with
respect to raw head outputs. A zero gradient indicates that a value is outside
the clamp interval and is blocked by `torch.clamp`; values inside the interval
retain the objective gradient.

```text
{gradients.to_string(index=False)}
```

## Interpretation

The existing objective has optimum `u*=log(d)`. The audit confirms whether the
observed raw outputs are actually clamp-saturated and whether clamp-induced
zero gradients explain the lack of learning. If raw values are not at the
clamp while reliability is uniform, clamp saturation is not the main cause;
the objective's common-distance optimum and the low-resolution regional
features are more plausible contributors. No model, objective, checkpoint, or
training run was changed.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw uncertainty outputs.")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Experiment_3_Uncertainty_best.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "uncertainty_raw_visualizations",
    )
    parser.add_argument(
        "--raw-statistics", type=Path,
        default=ROOT / "uncertainty_raw_statistics.csv",
    )
    parser.add_argument(
        "--region-statistics", type=Path,
        default=ROOT / "uncertainty_region_statistics.csv",
    )
    parser.add_argument(
        "--gradient-audit", type=Path,
        default=ROOT / "uncertainty_gradient_audit.csv",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "uncertainty_raw_audit_report.md",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset = SFTL54PairDataset(split="val")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmarks = batch["landmarks"].to(device)
            append_rows(rows, batch, capture(model, rgb, thermal, landmarks), "clean", 0.0)
            for region_index, region in enumerate(REGION_NAMES):
                for sigma in STRENGTHS:
                    corrupted_rgb = blur_region(rgb, landmarks, region_index, sigma)
                    corrupted_thermal = blur_region(thermal, landmarks, region_index, sigma)
                    append_rows(
                        rows,
                        batch,
                        capture(model, corrupted_rgb, corrupted_thermal, landmarks),
                        region,
                        sigma,
                    )
    report = pd.DataFrame(rows)
    statistics = stats_table(report[report["condition"] == "clean"])
    regions = region_table(report)
    gradients = gradient_table(report[report["condition"] == "clean"])
    report.to_csv(ROOT / "uncertainty_raw_samples.csv", index=False)
    statistics.to_csv(args.raw_statistics, index=False)
    regions.to_csv(args.region_statistics, index=False)
    gradients.to_csv(args.gradient_audit, index=False)
    write_visualizations(report, args.output_dir)
    write_report(report, statistics, regions, gradients, args.report)
    print(f"saved raw samples: {len(report)} rows")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
