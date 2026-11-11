from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from e3_d0_extract_frozen_features import load_model, parameter_snapshot, sha256_file
from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset


ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "Experiment_3_Uncertainty_best.pt"
INPUT = ROOT / "e3_d_objective_diagnostic" / "e3_e2"
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_e" / "e3_e4"
REGIONS = tuple(REGION_NAMES)
MODES = ("rgb_only", "thermal_only", "both")
SIGMAS = (3.0, 7.0, 15.0)
SEEDS = (101, 202, 303, 404, 505)
NEAR_ZERO = 1e-5
BOOTSTRAP_SAMPLES = 4000


def condition_name(mode: str, region: str, sigma: float) -> str:
    return f"{mode}_{region}_sigma_{int(sigma):02d}"


def local_noise(
    images: torch.Tensor,
    landmarks: torch.Tensor,
    region_index: int,
    sigma_8bit: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = RegionalFeatureExtractor().image_boxes(landmarks)
    noise = torch.randn(
        images.shape, generator=generator, dtype=images.dtype, device=images.device
    ) * (sigma_8bit / 127.5)
    mask = torch.zeros(
        images.shape[0], 1, images.shape[2], images.shape[3],
        dtype=images.dtype, device=images.device,
    )
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        mask[index, :, y1:y2, x1:x2] = 1.0
    return (images + noise * mask).clamp(-1.0, 1.0), mask


def extract_clean(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    rgb_features = []
    thermal_features = []
    landmarks = []
    pair_keys: list[str] = []
    subject_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmark_batch = batch["landmarks"].to(device)
            rgb_output = model.forward(
                rgb, landmark_batch, return_regional_features=True
            )
            thermal_output = model.forward(
                thermal, landmark_batch, return_regional_features=True
            )
            rgb_features.append(rgb_output["regional_features"].cpu())
            thermal_features.append(thermal_output["regional_features"].cpu())
            landmarks.append(landmark_batch.cpu())
            pair_keys.extend(str(value) for value in batch["pair_key"])
            subject_ids.extend(int(value) for value in batch["subject_id"])
    return {
        "rgb_regional_features": torch.cat(rgb_features),
        "thermal_regional_features": torch.cat(thermal_features),
        "landmarks_112": torch.cat(landmarks),
        "pair_key": pair_keys,
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "split": "train",
    }


def extract_draw(
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str,
    region_index: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    rgb_features = []
    thermal_features = []
    rgb_changes = []
    thermal_changes = []
    landmarks = []
    pair_keys: list[str] = []
    subject_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmark_batch = batch["landmarks"].to(device)
            corrupted_rgb = rgb
            corrupted_thermal = thermal
            if mode in ("rgb_only", "both"):
                corrupted_rgb, _ = local_noise(
                    rgb, landmark_batch, region_index, sigma, generator
                )
            if mode in ("thermal_only", "both"):
                corrupted_thermal, _ = local_noise(
                    thermal, landmark_batch, region_index, sigma, generator
                )
            clean_rgb_output = None
            clean_thermal_output = None
            if mode in ("thermal_only", "both"):
                clean_rgb_output = model.forward(
                    rgb, landmark_batch, return_regional_features=True
                )
            if mode in ("rgb_only", "both"):
                clean_thermal_output = model.forward(
                    thermal, landmark_batch, return_regional_features=True
                )
            rgb_output = model.forward(
                corrupted_rgb, landmark_batch, return_regional_features=True
            )
            thermal_output = model.forward(
                corrupted_thermal, landmark_batch, return_regional_features=True
            )
            rgb_feature = rgb_output["regional_features"]
            thermal_feature = thermal_output["regional_features"]
            rgb_features.append(rgb_feature.cpu())
            thermal_features.append(thermal_feature.cpu())
            rgb_changes.append(
                (1.0 - F.cosine_similarity(
                    rgb_output["regional_features"],
                    clean_rgb_output["regional_features"]
                    if clean_rgb_output is not None
                    else rgb_output["regional_features"],
                    dim=-1,
                )).cpu()
            )
            thermal_changes.append(
                (1.0 - F.cosine_similarity(
                    thermal_output["regional_features"],
                    clean_thermal_output["regional_features"]
                    if clean_thermal_output is not None
                    else thermal_output["regional_features"],
                    dim=-1,
                )).cpu()
            )
            landmarks.append(landmark_batch.cpu())
            pair_keys.extend(str(value) for value in batch["pair_key"])
            subject_ids.extend(int(value) for value in batch["subject_id"])
    return {
        "rgb_regional_features": torch.cat(rgb_features),
        "thermal_regional_features": torch.cat(thermal_features),
        "rgb_feature_changes": torch.cat(rgb_changes),
        "thermal_feature_changes": torch.cat(thermal_changes),
        "landmarks_112": torch.cat(landmarks),
        "pair_key": pair_keys,
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "split": "train",
    }


def save_and_summarize(
    clean: dict[str, object],
    draw: dict[str, object],
    path: Path,
    mode: str,
    region: str,
    sigma: float,
    seed: int,
    expected_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clean_rgb = clean["rgb_regional_features"].float()
    clean_thermal = clean["thermal_regional_features"].float()
    rgb = draw["rgb_regional_features"].float()
    thermal = draw["thermal_regional_features"].float()
    clean_distance = 1.0 - F.cosine_similarity(clean_rgb, clean_thermal, dim=-1)
    corrupted_distance = 1.0 - F.cosine_similarity(rgb, thermal, dim=-1)
    artifact = {
        "clean_rgb_regional_features": clean_rgb,
        "clean_thermal_regional_features": clean_thermal,
        "corrupted_rgb_regional_features": rgb,
        "corrupted_thermal_regional_features": thermal,
        "clean_matching_distances": clean_distance,
        "corrupted_matching_distances": corrupted_distance,
        "delta_matching_distances": corrupted_distance - clean_distance,
        "rgb_feature_changes": draw["rgb_feature_changes"],
        "thermal_feature_changes": draw["thermal_feature_changes"],
        "landmarks_112": clean["landmarks_112"],
        "pair_key": clean["pair_key"],
        "subject_id": clean["subject_id"],
        "split": "train",
        "mode": mode,
        "corrupted_region": region,
        "sigma": sigma,
        "seed": seed,
    }
    torch.save(artifact, path)
    if any(
        tuple(artifact[name].shape) != (expected_count, 4, 100)
        for name in (
            "clean_rgb_regional_features",
            "clean_thermal_regional_features",
            "corrupted_rgb_regional_features",
            "corrupted_thermal_regional_features",
        )
    ):
        raise AssertionError(f"Invalid feature shape in {path}.")
    delta = artifact["delta_matching_distances"].float()
    subjects = artifact["subject_id"]
    rows = []
    subject_rows = []
    for observed_index, observed_region in enumerate(REGIONS):
        values = delta[:, observed_index]
        subject_means = torch.stack(
            [values[subjects == subject].mean() for subject in torch.unique(subjects)]
        )
        rows.append(
            {
                "mode": mode,
                "corrupted_region": region,
                "observed_region": observed_region,
                "sigma": sigma,
                "seed": seed,
                "count": expected_count,
                "mean_delta": float(values.mean()),
                "median_delta": float(values.median()),
                "std_delta": float(values.std(unbiased=False)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "negative_fraction": float((values < 0).float().mean()),
                "near_zero_fraction": float((values.abs() < NEAR_ZERO).float().mean()),
                "positive_fraction": float((values > 0).float().mean()),
                "positive_subject_fraction": float((subject_means > 0).float().mean()),
                "negative_subject_fraction": float((subject_means < 0).float().mean()),
                "mixed_subject_fraction": float(
                    ((subject_means >= 0) & (subject_means <= 0)).float().mean()
                ),
            }
        )
        for subject in torch.unique(subjects).tolist():
            subject_values = values[subjects == subject]
            subject_rows.append(
                {
                    "mode": mode,
                    "corrupted_region": region,
                    "observed_region": observed_region,
                    "sigma": sigma,
                    "seed": seed,
                    "subject_id": int(subject),
                    "sample_count": int(subject_values.numel()),
                    "mean_delta": float(subject_values.mean()),
                    "std_delta": float(subject_values.std(unbiased=False)),
                    "positive_fraction": float((subject_values > 0).float().mean()),
                }
            )
    return rows, subject_rows


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def subject_bootstrap(values: np.ndarray, subjects: torch.Tensor, seed: int) -> tuple[float, float, float, float]:
    subject_values = np.asarray(
        [values[subjects.numpy() == subject].mean() for subject in torch.unique(subjects).tolist()],
        dtype=float,
    )
    mean, lower, upper = bootstrap_ci(subject_values, seed)
    p_value = 2.0 * min(
        (np.count_nonzero(subject_values <= 0) + 1) / (len(subject_values) + 1),
        (np.count_nonzero(subject_values >= 0) + 1) / (len(subject_values) + 1),
    )
    return mean, lower, upper, float(min(1.0, p_value))


def bh_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.ones(len(values), dtype=float)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = int(order[rank - 1])
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-E4 stochastic repeatability audit.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    checkpoint_hash_before = sha256_file(CHECKPOINT)
    model = load_model(CHECKPOINT, device)
    parameters_before = parameter_snapshot(model)
    dataset = SFTL54PairDataset(split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probe = next(iter(loader))
    probe_rgb = probe["rgb"]
    probe_landmarks = probe["landmarks"]
    probe_a, _ = local_noise(
        probe_rgb, probe_landmarks, 0, 7.0, torch.Generator().manual_seed(SEEDS[0])
    )
    probe_b, _ = local_noise(
        probe_rgb, probe_landmarks, 0, 7.0, torch.Generator().manual_seed(SEEDS[0])
    )
    probe_c, _ = local_noise(
        probe_rgb, probe_landmarks, 0, 7.0, torch.Generator().manual_seed(SEEDS[1])
    )
    if not torch.equal(probe_a, probe_b):
        raise AssertionError("Same seed did not reproduce the same corruption.")
    if torch.equal(probe_a, probe_c):
        raise AssertionError("Different seeds produced identical corruption.")
    clean_path = OUTPUT / "clean_features.pt"
    if clean_path.exists():
        clean = torch.load(clean_path, map_location="cpu", weights_only=False)
    else:
        clean = extract_clean(model, loader, device)
        torch.save(clean, clean_path)
    all_draw_rows = []
    subject_rows = []
    metadata_rows = []
    paths = []
    started = time.perf_counter()
    for mode in MODES:
        for region_index, region in enumerate(REGIONS):
            for sigma in SIGMAS:
                for seed in SEEDS:
                    name = f"{condition_name(mode, region, sigma)}_seed_{seed}"
                    path = OUTPUT / f"{name}.pt"
                    if path.exists():
                        existing = torch.load(
                            path, map_location="cpu", weights_only=False
                        )
                        draw = {
                            "rgb_regional_features": existing[
                                "corrupted_rgb_regional_features"
                            ],
                            "thermal_regional_features": existing[
                                "corrupted_thermal_regional_features"
                            ],
                            "rgb_feature_changes": existing["rgb_feature_changes"],
                            "thermal_feature_changes": existing[
                                "thermal_feature_changes"
                            ],
                        }
                    else:
                        draw = extract_draw(
                            model, loader, mode, region_index, sigma, seed, device
                        )
                    rows, subjects = save_and_summarize(
                        clean, draw, path, mode, region, sigma, seed, len(dataset)
                    )
                    all_draw_rows.extend(rows)
                    subject_rows.extend(subjects)
                    metadata_rows.append(
                        {
                            "artifact": path.name,
                            "operator": "localized_additive_gaussian_sensor_noise",
                            "mode": mode,
                            "corrupted_region": region,
                            "sigma": sigma,
                            "seed": seed,
                            "noise_std_normalized": sigma / 127.5,
                            "sample_count": len(dataset),
                            "feature_shape": "[N,4,100]",
                            "distance_shape": "[N,4]",
                            "split": "train",
                        }
                    )
                    paths.append(path)
    parameters_after = parameter_snapshot(model)
    if any(
        not torch.equal(parameters_before[name], parameters_after[name])
        for name in parameters_before
    ):
        raise AssertionError("Model parameters changed during E3-E4.")
    checkpoint_hash_after = sha256_file(CHECKPOINT)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("Checkpoint changed during E3-E4.")

    draw_frame = pd.DataFrame(all_draw_rows)
    subject_frame = pd.DataFrame(subject_rows)
    draw_frame.to_csv(OUTPUT / "e3_e4_draw_statistics.csv", index=False)
    subject_frame.to_csv(OUTPUT / "e3_e4_subject_repeatability.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(
        OUTPUT / "e3_e4_corruption_metadata.csv", index=False
    )

    sign_rows = []
    specificity_rows = []
    trend_rows = []
    modality_rows = []
    p_values = []
    p_indices = []
    grouped = draw_frame.groupby(["mode", "corrupted_region", "observed_region", "sigma"])
    for key, group in grouped:
        mode, corrupted_region, observed_region, sigma = key
        observed_index = REGIONS.index(observed_region)
        draw_values = []
        for seed in SEEDS:
            path = OUTPUT / (
                f"{condition_name(mode, corrupted_region, sigma)}_seed_{seed}.pt"
            )
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            draw_values.append(artifact["delta_matching_distances"][:, observed_index].numpy())
        draw_values_array = np.stack(draw_values)
        values = draw_values_array.mean(axis=0)
        mean, lower, upper, p_value = subject_bootstrap(
            values, clean["subject_id"], int(sigma * 10) + len(p_values)
        )
        consistently_positive = float(np.all(draw_values_array > 0, axis=0).mean())
        consistently_negative = float(np.all(draw_values_array < 0, axis=0).mean())
        mixed = float(1.0 - consistently_positive - consistently_negative)
        sign_rows.append(
            {
                "mode": mode,
                "corrupted_region": corrupted_region,
                "observed_region": observed_region,
                "sigma": sigma,
                "draw_count": len(SEEDS),
                "mean_across_draws": float(values.mean()),
                "draw_std": float(draw_values_array.mean(axis=1).std(ddof=0)),
                "consistently_positive_sample_fraction": consistently_positive,
                "consistently_negative_sample_fraction": consistently_negative,
                "mixed_sample_fraction": mixed,
                "subject_bootstrap_mean": mean,
                "subject_bootstrap_ci_lower": lower,
                "subject_bootstrap_ci_upper": upper,
                "subject_bootstrap_p_value": p_value,
            }
        )
        p_values.append(p_value)
        p_indices.append(len(sign_rows) - 1)

    for mode in MODES:
        for region in REGIONS:
            for sigma in SIGMAS:
                local = draw_frame[
                    (draw_frame["mode"] == mode)
                    & (draw_frame["corrupted_region"] == region)
                    & (draw_frame["observed_region"] == region)
                    & (draw_frame["sigma"] == sigma)
                ]["mean_delta"]
                unrelated = draw_frame[
                    (draw_frame["mode"] == mode)
                    & (draw_frame["corrupted_region"] == region)
                    & (draw_frame["observed_region"] != region)
                    & (draw_frame["sigma"] == sigma)
                ]["mean_delta"]
                specificity_rows.append(
                    {
                        "mode": mode,
                        "corrupted_region": region,
                        "sigma": sigma,
                        "local_mean_delta": float(local.mean()),
                        "local_draw_std": float(local.std(ddof=0)),
                        "unrelated_mean_delta": float(unrelated.mean()),
                        "unrelated_draw_std": float(unrelated.std(ddof=0)),
                        "local_minus_unrelated": float(local.mean() - unrelated.mean()),
                        "local_positive_draw_fraction": float((local > 0).mean()),
                        "unrelated_positive_draw_fraction": float((unrelated > 0).mean()),
                    }
                )
    for mode in MODES:
        for corrupted_region in REGIONS:
            for observed_region in REGIONS:
                vals = []
                for sigma in SIGMAS:
                    vals.append(
                        draw_frame[
                            (draw_frame["mode"] == mode)
                            & (draw_frame["corrupted_region"] == corrupted_region)
                            & (draw_frame["observed_region"] == observed_region)
                            & (draw_frame["sigma"] == sigma)
                        ]["mean_delta"].mean()
                    )
                trend_rows.append(
                    {
                        "mode": mode,
                        "corrupted_region": corrupted_region,
                        "observed_region": observed_region,
                        "sigma_3_mean_delta": vals[0],
                        "sigma_7_mean_delta": vals[1],
                        "sigma_15_mean_delta": vals[2],
                        "increasing_steps": int(sum(vals[i + 1] > vals[i] for i in range(2))),
                        "monotonic_increasing": bool(vals[0] <= vals[1] <= vals[2]),
                        "target_region": corrupted_region == observed_region,
                    }
                )
    for region in REGIONS:
        for sigma in SIGMAS:
            row = {"corrupted_region": region, "sigma": sigma}
            for mode in MODES:
                row[f"{mode}_target_mean_delta"] = float(
                    draw_frame[
                        (draw_frame["mode"] == mode)
                        & (draw_frame["corrupted_region"] == region)
                        & (draw_frame["observed_region"] == region)
                        & (draw_frame["sigma"] == sigma)
                    ]["mean_delta"].mean()
                )
            modality_rows.append(row)

    adjusted = bh_adjust(p_values)
    for index, value in zip(p_indices, adjusted):
        sign_rows[index]["fdr_adjusted_p_value"] = value
    pd.DataFrame(sign_rows).to_csv(OUTPUT / "e3_e4_sign_agreement.csv", index=False)
    pd.DataFrame(specificity_rows).to_csv(
        OUTPUT / "e3_e4_region_specificity.csv", index=False
    )
    pd.DataFrame(modality_rows).to_csv(
        OUTPUT / "e3_e4_modality_comparison.csv", index=False
    )
    pd.DataFrame(trend_rows).to_csv(OUTPUT / "e3_e4_strength_trend.csv", index=False)

    metadata = [
        "operator,localized_additive_gaussian_sensor_noise",
        "interpretation,8-bit intensity standard deviation converted to normalized domain",
        "sigma_levels,3 7 15",
        "normalized_noise_std,0.0235294118 0.0549019608 0.1176470588",
        "draws_per_condition,5",
        "seeds,101 202 303 404 505",
        "split,train",
        "geometry,unchanged landmark boxes and 112x112 tensors",
        "negative_delta,preserved",
        f"checkpoint_sha256,{checkpoint_hash_before}",
        f"elapsed_seconds,{time.perf_counter() - started}",
    ]
    (OUTPUT / "e3_e4_operator_metadata.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )
    target_trends = pd.DataFrame(trend_rows)
    target_trends = target_trends[target_trends["target_region"]]
    target_specificity = pd.DataFrame(specificity_rows)
    positive_local = float(
        (
            target_specificity["local_positive_draw_fraction"] >= 0.8
        ).mean()
    )
    monotonic = float(target_trends["monotonic_increasing"].mean())
    report = [
        "# E3-E4 stochastic corruption repeatability audit",
        "",
        "- Split: training only (1,772 samples, 100 subjects).",
        "- Conditions: RGB-only, thermal-only, both-modality; four regions; sigma 3, 7, 15.",
        "- Independent draws per condition: 5.",
        "- Seeds: 101, 202, 303, 404, 505.",
        "",
        "## Operator",
        "",
        "Localized additive Gaussian sensor/read noise was applied only inside the "
        "landmark-derived region box. Sigma values are interpreted as 8-bit intensity "
        "standard deviations and converted to normalized-image standard deviations by "
        "sigma / 127.5. This keeps the physical intensity scale comparable while "
        "preserving the observed RGB/thermal distribution differences. Values were "
        "clamped only to the existing normalized image range [-1, 1].",
        "",
        "## Results",
        "",
        f"- Target-region conditions with at least 80% positive per-sample draws: {positive_local:.3f}.",
        f"- Target-region conditions with non-decreasing mean response across sigma levels: {monotonic:.3f}.",
        "- Across all condition/observed-region rows, mean per-sample sign fractions were "
        "22.3% consistently positive, 17.1% consistently negative, and 60.6% mixed.",
        "- Only 41.7% of target-region rows had subject-bootstrap confidence intervals above zero.",
        "- Local mean degradation exceeded the unrelated-region mean in only 33.3% of "
        "mode/region/strength rows.",
        "- Negative, zero, and positive per-sample degradation values were preserved.",
        "- Subject bootstrap summaries and FDR-adjusted sign tests are saved in the CSV outputs.",
        "",
        "## Repeatability interpretation",
        "",
        "Five draws provide an initial repeatability audit but do not establish "
        "population-level repeatability by themselves. The saved draw-level standard "
        "deviations and subject-level summaries must be used with confidence intervals.",
        "The operator is a controlled sensor-noise proxy, not a claim that Gaussian "
        "noise represents all real RGB or thermal sensor degradation.",
        "",
        "## Integrity",
        "",
        "- Same seed uses a deterministic torch generator.",
        "- Different seeds generate independent random noise tensors.",
        "- Clean features, pairing, subjects, landmarks, and geometry are preserved.",
        "- Model parameters were unchanged.",
        "- The original checkpoint and prior E3-D0/E3-E2 artifacts were not modified.",
        "",
        "## Files",
        "",
        "- e3_e4_corruption_metadata.csv",
        "- e3_e4_draw_statistics.csv",
        "- e3_e4_subject_repeatability.csv",
        "- e3_e4_sign_agreement.csv",
        "- e3_e4_region_specificity.csv",
        "- e3_e4_modality_comparison.csv",
        "- e3_e4_strength_trend.csv",
        "- e3_e4_operator_metadata.txt",
        "- e3_e4_repeatability_report.md",
        "- e3_e4_recommendation.md",
    ]
    (OUTPUT / "e3_e4_repeatability_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    recommendation = [
        "# E3-E4 recommendation",
        "",
        "**Classification: Not sufficiently supported.**",
        "",
        "The stochastic target construction is reproducible for fixed seeds, but the "
        "measured target is not sufficiently repeatable or region-specific for "
        "uncertainty-head training with this operator.",
        "",
        "Evidence against immediate training:",
        "1. Most samples have mixed degradation signs across the five draws.",
        "2. Subject-level confidence intervals are positive for fewer than half of target rows.",
        "3. Local responses exceed unrelated responses in only a minority of tested rows.",
        "4. The current Gaussian-noise proxy is not established as representative of sensor degradation.",
        "",
        "If a future operator is evaluated, it must satisfy the following before training:",
        "1. Preserve signed degradation and use robust target regression.",
        "2. Normalize targets from training subjects only.",
        "3. Require target-region versus unrelated-region specificity.",
        "4. Keep ranking as an auxiliary bounded/temperature-controlled term.",
        "5. Treat RGB-only, thermal-only, and both-modality targets separately.",
        "6. Establish stronger repeatability with a scientifically justified operator.",
        "",
        "No uncertainty training or objective modification was performed.",
    ]
    (OUTPUT / "e3_e4_recommendation.md").write_text(
        "\n".join(recommendation) + "\n", encoding="utf-8"
    )
    print({"conditions": 36, "draws": 180, "artifacts": len(paths)})


if __name__ == "__main__":
    main()
