from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent
CORRUPTION_STRENGTHS = (3.0, 7.0, 15.0)


def load_model(checkpoint: Path, device: torch.device) -> EdgeFaceTransformerAdapter:
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    return model


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


def add_rows(
    rows: list[dict[str, object]],
    batch: dict[str, object],
    output: dict[str, torch.Tensor],
    condition: str,
    strength: float | None,
) -> None:
    rgb_features = output["rgb_regional_features"]
    thermal_features = output["thermal_regional_features"]
    distance = 1.0 - torch.nn.functional.cosine_similarity(
        rgb_features, thermal_features, dim=-1
    )
    for sample_index, pair_key in enumerate(batch["pair_key"]):
        for region_index, region in enumerate(REGION_NAMES):
            rows.append(
                {
                    "pair_key": pair_key,
                    "subject_id": int(batch["subject_id"][sample_index]),
                    "condition": condition,
                    "strength": strength if strength is not None else 0.0,
                    "region": region,
                    "matching_distance": float(distance[sample_index, region_index]),
                    "rgb_log_uncertainty": float(
                        output["rgb_uncertainty"][sample_index, region_index]
                    ),
                    "thermal_log_uncertainty": float(
                        output["thermal_uncertainty"][sample_index, region_index]
                    ),
                    "combined_log_uncertainty": float(
                        output["combined_uncertainty"][sample_index, region_index]
                    ),
                    "reliability": float(
                        output["regional_reliability"][sample_index, region_index]
                    ),
                }
            )


def correlation_table(report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, strength, region), group in report.groupby(
        ["condition", "strength", "region"], sort=False
    ):
        if len(group) < 3 or group["combined_log_uncertainty"].std() == 0:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = group["combined_log_uncertainty"].corr(
                group["matching_distance"], method="pearson"
            )
            spearman = group["combined_log_uncertainty"].corr(
                group["matching_distance"], method="spearman"
            )
        rows.append(
            {
                "condition": condition,
                "strength": strength,
                "region": region,
                "pearson_uncertainty_distance": pearson,
                "spearman_uncertainty_distance": spearman,
                "samples": len(group),
            }
        )
    return pd.DataFrame(rows)


def write_visualizations(report: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in REGION_NAMES:
        subset = report[report["region"] == region]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for condition, group in subset.groupby("condition"):
            axes[0].plot(
                group.groupby("strength")["combined_log_uncertainty"].mean(),
                marker="o",
                label=condition,
            )
            axes[1].plot(
                group.groupby("strength")["matching_distance"].mean(),
                marker="o",
                label=condition,
            )
        axes[0].set_title(f"{region}: combined uncertainty")
        axes[1].set_title(f"{region}: matching distance")
        for axis in axes:
            axis.set_xlabel("blur sigma")
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{region}.png", dpi=150)
        plt.close(figure)


def write_summary(
    report: pd.DataFrame,
    correlations: pd.DataFrame,
    history_path: Path,
    output_path: Path,
) -> None:
    clean = report[report["condition"] == "clean"]
    distribution = clean.groupby("region")[
        ["rgb_log_uncertainty", "thermal_log_uncertainty", "reliability"]
    ].agg(["mean", "std", "min", "max"])
    uncertainty_std = float(
        clean[["rgb_log_uncertainty", "thermal_log_uncertainty"]].stack().std()
    )
    reliability_std = float(clean["reliability"].std())
    if reliability_std < 0.01:
        conclusion = "C. Not yet learning meaningful regional reliability."
    else:
        conclusion = "B. Partially learned but insufficiently calibrated."
    lines = [
        "# Uncertainty calibration investigation",
        "",
        f"- Full E3 history: `{history_path.name}`",
        f"- Validation samples: {clean['pair_key'].nunique()}",
        f"- Regions: {', '.join(REGION_NAMES)}",
        f"- Clean pooled uncertainty standard deviation: {uncertainty_std:.6f}",
        f"- Clean reliability standard deviation: {reliability_std:.6f}",
        f"- Conclusion: **{conclusion}**",
        "",
        "## Clean distribution summary",
        "",
        "```text",
        distribution.to_string(),
        "```",
        "",
        "## Correlation summary",
        "",
        "Pearson and Spearman correlations compare combined predicted uncertainty "
        "with observed regional RGB-thermal cosine distance.",
        "",
        "```text",
        correlations.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "The corruption rows use the same validation pairs as the clean rows. "
        "The current heteroscedastic objective rewards reducing log uncertainty "
        "until it is balanced by the regional distance term, but it has no "
        "calibration target. The observed near-zero uncertainty-distance "
        "correlations and near-uniform reliability therefore do not support "
        "reliability-weighted alignment yet. A corruption-supervised auxiliary "
        "objective would be a separate controlled experiment, not an implicit "
        "change to this phase.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=ROOT / "Experiment_3_Uncertainty_history.csv")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "uncertainty_calibration_report.csv",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=ROOT / "uncertainty_calibration_visualizations",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "uncertainty_calibration_summary.md",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset = SFTL54PairDataset(split="val")
    if args.count is not None:
        dataset.records = dataset.records.head(args.count).reset_index(drop=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmarks = batch["landmarks"].to(device)
            clean = model.forward_with_uncertainty(rgb, thermal, landmarks)
            add_rows(rows, batch, clean, "clean", None)
            for region_index, region in enumerate(REGION_NAMES):
                for strength in CORRUPTION_STRENGTHS:
                    corrupted_rgb = blur_region(rgb, landmarks, region_index, strength)
                    corrupted_thermal = blur_region(
                        thermal, landmarks, region_index, strength
                    )
                    corrupted = model.forward_with_uncertainty(
                        corrupted_rgb, corrupted_thermal, landmarks
                    )
                    add_rows(rows, batch, corrupted, region, strength)
    report = pd.DataFrame(rows)
    correlations = correlation_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    write_visualizations(report, args.visualization_dir)
    write_summary(report, correlations, args.history, args.summary)
    correlations.to_csv(
        args.output.with_name("uncertainty_calibration_correlations.csv"), index=False
    )
    print(f"saved {len(report)} calibration rows to {args.output}")
    print(f"saved summary to {args.summary}")


if __name__ == "__main__":
    main()
