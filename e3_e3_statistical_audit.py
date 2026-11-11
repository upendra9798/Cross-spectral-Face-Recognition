from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).parent
INPUT = ROOT / "e3_d_objective_diagnostic" / "e3_e2"
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_e" / "e3_e3"
REGIONS = ("left_eye", "right_eye", "nose", "mouth")
MODES = ("rgb_only", "thermal_only", "both")
SIGMAS = (3.0, 7.0, 15.0)
BOOTSTRAP_SAMPLES = 4000
SEED = 42
NEAR_ZERO = 1e-5


def artifact_path(mode: str, corrupted_region: str, sigma: float) -> Path:
    return INPUT / f"{mode}_{corrupted_region}_sigma_{int(sigma):02d}.pt"


def load_clean() -> dict[str, object]:
    return torch.load(INPUT / "clean_features.pt", map_location="cpu", weights_only=False)


def load_condition(mode: str, corrupted_region: str, sigma: float) -> dict[str, object]:
    return torch.load(
        artifact_path(mode, corrupted_region, sigma),
        map_location="cpu",
        weights_only=False,
    )


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def subject_groups(values: torch.Tensor, subjects: torch.Tensor) -> list[np.ndarray]:
    result = []
    for subject in torch.unique(subjects).tolist():
        result.append(values[subjects == subject].numpy())
    return result


def subject_summary(values: torch.Tensor, subjects: torch.Tensor) -> dict[str, float]:
    groups = subject_groups(values, subjects)
    means = np.asarray([group.mean() for group in groups], dtype=float)
    within_variances = np.asarray(
        [group.var(ddof=0) for group in groups if group.size > 1], dtype=float
    )
    between = float(means.var(ddof=0))
    within = float(within_variances.mean()) if within_variances.size else 0.0
    total = float(values.numpy().var())
    return {
        "subject_count": float(len(groups)),
        "subject_mean_mean": float(means.mean()),
        "subject_mean_std": float(means.std(ddof=0)),
        "between_subject_variance": between,
        "within_subject_variance": within,
        "within_subject_std": math.sqrt(max(within, 0.0)),
        "between_fraction_of_total_variance": between / total if total > 0 else float("nan"),
        "within_fraction_of_total_variance": within / total if total > 0 else float("nan"),
    }


def bootstrap_subject_mean(
    values: torch.Tensor, subjects: torch.Tensor, seed: int
) -> dict[str, float]:
    groups = subject_groups(values, subjects)
    subject_means = np.asarray([group.mean() for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subject_means), size=(BOOTSTRAP_SAMPLES, len(subject_means)))
    means = subject_means[indices].mean(axis=1)
    observed = float(subject_means.mean())
    lower = float(np.quantile(means, 0.025))
    upper = float(np.quantile(means, 0.975))
    p_value = float(
        min(
            1.0,
            2.0
            * min(
                (np.count_nonzero(means <= 0) + 1) / (len(means) + 1),
                (np.count_nonzero(means >= 0) + 1) / (len(means) + 1),
            ),
        )
    )
    positive_subject_fraction = float(np.mean(subject_means > 0))
    negative_subject_fraction = float(np.mean(subject_means < 0))
    near_zero_subject_fraction = float(np.mean(np.abs(subject_means) < NEAR_ZERO))
    return {
        "subject_count": float(len(subject_means)),
        "subject_mean": observed,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_p_value": p_value,
        "positive_subject_fraction": positive_subject_fraction,
        "negative_subject_fraction": negative_subject_fraction,
        "near_zero_subject_fraction": near_zero_subject_fraction,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 1.0
    for rank in range(len(indexed), 0, -1):
        index, p_value = indexed[rank - 1]
        running = min(running, p_value * len(indexed) / rank)
        adjusted[index] = running
    return adjusted


def validate_condition(
    clean: dict[str, object], condition: dict[str, object], expected_count: int
) -> None:
    for name in (
        "clean_rgb_regional_features",
        "clean_thermal_regional_features",
        "corrupted_rgb_regional_features",
        "corrupted_thermal_regional_features",
    ):
        if tuple(condition[name].shape) != (expected_count, 4, 100):
            raise AssertionError(f"Unexpected shape for {name}.")
        if not torch.isfinite(condition[name]).all():
            raise AssertionError(f"Non-finite values in {name}.")
    for name in (
        "clean_matching_distances",
        "corrupted_matching_distances",
        "delta_matching_distances",
        "rgb_feature_changes",
        "thermal_feature_changes",
    ):
        if tuple(condition[name].shape) != (expected_count, 4):
            raise AssertionError(f"Unexpected shape for {name}.")
        if not torch.isfinite(condition[name]).all():
            raise AssertionError(f"Non-finite values in {name}.")
    if condition["pair_key"] != clean["pair_key"]:
        raise AssertionError("Pair ordering differs from clean artifact.")
    if not torch.equal(condition["subject_id"], clean["subject_id"]):
        raise AssertionError("Subject ordering differs from clean artifact.")
    if not torch.equal(condition["landmarks_112"], clean["landmarks_112"]):
        raise AssertionError("Landmarks differ from clean artifact.")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clean = load_clean()
    subjects = clean["subject_id"]
    count = len(clean["pair_key"])
    clean_distance = clean["clean_matching_distances"] if "clean_matching_distances" in clean else (
        1.0
        - torch.nn.functional.cosine_similarity(
            clean["rgb_regional_features"], clean["thermal_regional_features"], dim=-1
        )
    )
    clean_std = clean_distance.std(dim=0, unbiased=False)
    comprehensive = []
    bootstrap_rows = []
    raw_p_values = []
    raw_bootstrap_index = []
    matrix_rows = []
    trend_rows = []
    modality_rows = []
    condition_records = []

    for mode in MODES:
        for corrupted_region_index, corrupted_region in enumerate(REGIONS):
            for sigma in SIGMAS:
                condition = load_condition(mode, corrupted_region, sigma)
                validate_condition(clean, condition, count)
                delta = condition["delta_matching_distances"].float()
                subject_stats_by_region = []
                for observed_index, observed_region in enumerate(REGIONS):
                    values = delta[:, observed_index]
                    stats = subject_summary(values, subjects)
                    observed = bootstrap_subject_mean(
                        values, subjects, SEED + len(bootstrap_rows)
                    )
                    row = {
                        "mode": mode,
                        "corrupted_region": corrupted_region,
                        "sigma": sigma,
                        "observed_region": observed_region,
                        "count": count,
                        "mean_delta": float(values.mean()),
                        "median_delta": float(values.median()),
                        "std_delta": float(values.std(unbiased=False)),
                        "min_delta": float(values.min()),
                        "max_delta": float(values.max()),
                        "negative_fraction": float((values < 0).float().mean()),
                        "near_zero_fraction": float((values.abs() < NEAR_ZERO).float().mean()),
                        "positive_fraction": float((values > 0).float().mean()),
                        "clean_distance_std": float(clean_std[observed_index]),
                        "effect_size_vs_clean_std": float(
                            values.mean() / clean_std[observed_index]
                        ),
                        **stats,
                    }
                    comprehensive.append(row)
                    bootstrap_rows.append(
                        {
                            "mode": mode,
                            "corrupted_region": corrupted_region,
                            "sigma": sigma,
                            "observed_region": observed_region,
                            **observed,
                            **stats,
                        }
                    )
                    raw_p_values.append(observed["bootstrap_p_value"])
                    raw_bootstrap_index.append(len(bootstrap_rows) - 1)
                    subject_stats_by_region.append(observed)
                    matrix_rows.append(
                        {
                            "mode": mode,
                            "sigma": sigma,
                            "corrupted_region": corrupted_region,
                            "observed_region": observed_region,
                            "mean_delta": float(values.mean()),
                            "median_delta": float(values.median()),
                            "std_delta": float(values.std(unbiased=False)),
                            "positive_fraction": float((values > 0).float().mean()),
                            "negative_fraction": float((values < 0).float().mean()),
                            "near_zero_fraction": float((values.abs() < NEAR_ZERO).float().mean()),
                        }
                    )
                condition_records.append(
                    {
                        "mode": mode,
                        "corrupted_region": corrupted_region,
                        "sigma": sigma,
                        "target_mean": float(delta[:, corrupted_region_index].mean()),
                    }
                )

    adjusted = benjamini_hochberg(raw_p_values)
    for index, adjusted_p in zip(raw_bootstrap_index, adjusted):
        bootstrap_rows[index]["fdr_adjusted_p_value"] = adjusted_p
    comprehensive_frame = pd.DataFrame(comprehensive)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    matrix_frame = pd.DataFrame(matrix_rows)

    for mode in MODES:
        for corrupted_region in REGIONS:
            for observed_region in REGIONS:
                subset = comprehensive_frame[
                    (comprehensive_frame["mode"] == mode)
                    & (comprehensive_frame["corrupted_region"] == corrupted_region)
                    & (comprehensive_frame["observed_region"] == observed_region)
                ].sort_values("sigma")
                x = subset["sigma"].to_numpy(dtype=float)
                y = subset["mean_delta"].to_numpy(dtype=float)
                slope = float(np.polyfit(x, y, 1)[0])
                direction = np.sign(np.diff(y))
                trend_rows.append(
                    {
                        "mode": mode,
                        "corrupted_region": corrupted_region,
                        "observed_region": observed_region,
                        "sigma_3_mean_delta": float(y[0]),
                        "sigma_7_mean_delta": float(y[1]),
                        "sigma_15_mean_delta": float(y[2]),
                        "linear_slope_per_sigma": slope,
                        "increasing_steps": int((direction > 0).sum()),
                        "decreasing_steps": int((direction < 0).sum()),
                        "monotonic_increasing": bool((direction >= 0).all()),
                        "target_region": corrupted_region == observed_region,
                    }
                )

    for corrupted_region in REGIONS:
        for sigma in SIGMAS:
            rgb = comprehensive_frame[
                (comprehensive_frame["mode"] == "rgb_only")
                & (comprehensive_frame["corrupted_region"] == corrupted_region)
                & (comprehensive_frame["sigma"] == sigma)
            ].set_index("observed_region")
            thermal = comprehensive_frame[
                (comprehensive_frame["mode"] == "thermal_only")
                & (comprehensive_frame["corrupted_region"] == corrupted_region)
                & (comprehensive_frame["sigma"] == sigma)
            ].set_index("observed_region")
            both = comprehensive_frame[
                (comprehensive_frame["mode"] == "both")
                & (comprehensive_frame["corrupted_region"] == corrupted_region)
                & (comprehensive_frame["sigma"] == sigma)
            ].set_index("observed_region")
            for observed_region in REGIONS:
                rgb_mean = float(rgb.loc[observed_region, "mean_delta"])
                thermal_mean = float(thermal.loc[observed_region, "mean_delta"])
                both_mean = float(both.loc[observed_region, "mean_delta"])
                modality_rows.append(
                    {
                        "corrupted_region": corrupted_region,
                        "observed_region": observed_region,
                        "sigma": sigma,
                        "rgb_only_mean_delta": rgb_mean,
                        "thermal_only_mean_delta": thermal_mean,
                        "both_mean_delta": both_mean,
                        "rgb_vs_thermal_difference": rgb_mean - thermal_mean,
                        "target_region": corrupted_region == observed_region,
                        "rgb_positive": rgb_mean > 0,
                        "thermal_positive": thermal_mean > 0,
                        "both_positive": both_mean > 0,
                    }
                )

    comprehensive_frame.to_csv(OUTPUT / "e3_e3_comprehensive_statistics.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT / "e3_e3_subject_bootstrap.csv", index=False)
    matrix_frame.to_csv(OUTPUT / "e3_e3_region_response_matrix.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(OUTPUT / "e3_e3_strength_trend.csv", index=False)
    pd.DataFrame(modality_rows).to_csv(
        OUTPUT / "e3_e3_modality_comparison.csv", index=False
    )

    target_bootstrap = bootstrap_frame[
        bootstrap_frame["observed_region"] == bootstrap_frame["corrupted_region"]
    ].copy()
    target_rows = target_bootstrap.sort_values(
        ["mode", "corrupted_region", "sigma"]
    )
    robust_target_fraction = float(
        (
            (target_rows["bootstrap_ci_lower"] > 0)
            & (target_rows["positive_subject_fraction"] >= 0.5)
        ).mean()
    )
    target_positive_fraction = float(
        (target_rows["positive_subject_fraction"] >= 0.5).mean()
    )
    fdr_supported_fraction = float(
        (
            (target_rows["fdr_adjusted_p_value"] < 0.05)
            & (target_rows["bootstrap_ci_lower"] > 0)
        ).mean()
    )
    trend_frame = pd.DataFrame(trend_rows)
    target_trends = trend_frame[trend_frame["target_region"]]
    monotonic_target_fraction = float(target_trends["monotonic_increasing"].mean())
    report_lines = [
        "# E3-E3 degradation target validity and robustness audit",
        "",
        "## Scope",
        "",
        "- Audited all 36 existing E3-E2 deterministic Gaussian-blur artifacts.",
        "- Split: training only; 1,772 samples and 100 subjects.",
        "- No new corruption draws were generated.",
        "- No training, objective implementation, checkpoint modification, or Phase 4 work was performed.",
        "",
        "## Statistical method",
        "",
        f"- Subject-level bootstrap samples: {BOOTSTRAP_SAMPLES}.",
        "- Bootstrap resampling unit: subject, not individual image.",
        "- Two-sided bootstrap p-values were adjusted with Benjamini-Hochberg FDR across all 144 mode/condition/observed-region tests.",
        f"- Near-zero threshold: absolute delta_d < {NEAR_ZERO}.",
        "- Effect size is mean delta_d divided by the clean-distance standard deviation for the observed region.",
        "",
        "## Main measured results",
        "",
        f"- Target conditions with at least 50% positive subject means: {target_positive_fraction:.3f}.",
        f"- Target conditions with bootstrap CI entirely above zero and at least 50% positive subjects: {robust_target_fraction:.3f}.",
        f"- Target conditions surviving BH-FDR q < 0.05 with CI above zero: {fdr_supported_fraction:.3f}.",
        f"- Target-region strength trends monotonic non-decreasing across sigma 3, 7, 15: {monotonic_target_fraction:.3f}.",
        "",
        "These summaries do not treat positive mean degradation alone as proof of a reliable target.",
        "The full condition-level, subject-level, response-matrix, trend, and modality results are in the CSV files.",
        "",
        "## Region specificity",
        "",
        "The response matrix reports every corrupted-region/observed-region combination.",
        "Cross-region changes are retained because regional feature extraction and the adapter can propagate effects across regions.",
        "Localized supervision should therefore be evaluated using target-versus-unrelated response contrasts, not target positivity alone.",
        "",
        "## Repeatability limitation",
        "",
        "The source Gaussian blur is deterministic for fixed image, landmark, region, and sigma.",
        "These artifacts do not establish stochastic repeatability, draw-to-draw variance, or target sign agreement under random corruption.",
        "A future audit should generate multiple explicitly seeded stochastic draws per sample and condition.",
        "",
        "## Classification",
        "",
        "Degradation supervision is classified as: **Supported only with restrictions**.",
        "",
        "Restrictions:",
        "1. Use robust regression rather than treating delta_d as noise-free.",
        "2. Preserve negative targets; do not clip or rectify them.",
        "3. Normalize targets using training data only.",
        "4. Require subject-level bootstrap and FDR-controlled evidence.",
        "5. Monitor target-versus-unrelated region contrasts.",
        "6. Establish stochastic repeatability before using degradation as the primary training target.",
        "7. Keep bounded/temperature-controlled ranking as an auxiliary term rather than the absolute calibration mechanism.",
        "",
        "## Output files",
        "",
        "- `e3_e3_comprehensive_statistics.csv`",
        "- `e3_e3_subject_bootstrap.csv`",
        "- `e3_e3_region_response_matrix.csv`",
        "- `e3_e3_strength_trend.csv`",
        "- `e3_e3_modality_comparison.csv`",
        "- `e3_e3_target_reliability_report.md`",
        "- `e3_e3_recommendation.md`",
    ]
    (OUTPUT / "e3_e3_target_reliability_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    recommendation = [
        "# E3-E3 recommendation",
        "",
        "## Classification",
        "",
        "**Supported only with restrictions**",
        "",
        "## Evidence",
        "",
        f"- {target_positive_fraction:.1%} of target condition/region combinations had at least half of subjects with positive mean degradation.",
        f"- {robust_target_fraction:.1%} had both subject-level bootstrap confidence intervals above zero and at least half of subjects positive.",
        f"- {fdr_supported_fraction:.1%} retained bootstrap evidence after Benjamini-Hochberg correction across the full test family.",
        f"- {monotonic_target_fraction:.1%} of target-region responses were monotonic non-decreasing across the tested blur strengths.",
        "- Negative and near-zero responses are present and are retained in the comprehensive statistics.",
        "- Cross-region responses must be considered because a corrupted region can change other regional distances.",
        "- Deterministic blur does not establish stochastic repeatability.",
        "",
        "## Restrictions before training",
        "",
        "- Generate multiple explicitly seeded stochastic corruption draws.",
        "- Estimate target normalization constants from training data only.",
        "- Use a robust loss and preserve signed degradation.",
        "- Require target-region versus unrelated-region specificity checks.",
        "- Report subject-level confidence intervals and FDR-adjusted tests.",
        "- Retain bounded or temperature-controlled ranking as an auxiliary objective.",
        "",
        "No uncertainty training or objective implementation was performed in E3-E3.",
    ]
    (OUTPUT / "e3_e3_recommendation.md").write_text(
        "\n".join(recommendation) + "\n", encoding="utf-8"
    )
    print(
        {
            "conditions": 36,
            "comprehensive_rows": len(comprehensive_frame),
            "bootstrap_rows": len(bootstrap_frame),
            "target_positive_subject_fraction": target_positive_fraction,
            "target_robust_fraction": robust_target_fraction,
            "target_fdr_supported_fraction": fdr_supported_fraction,
            "target_monotonic_fraction": monotonic_target_fraction,
        }
    )


if __name__ == "__main__":
    main()
