from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from e3_d0_extract_frozen_features import (
    load_model,
    parameter_snapshot,
    sha256_file,
)
from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset


ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "Experiment_3_Uncertainty_best.pt"
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_e2"
REGIONS = tuple(REGION_NAMES)
SIGMAS = (3.0, 7.0, 15.0)
MODES = ("rgb_only", "thermal_only", "both")


def blur_region(
    images: torch.Tensor,
    landmarks: torch.Tensor,
    region_index: int,
    sigma: float,
) -> torch.Tensor:
    boxes = RegionalFeatureExtractor().image_boxes(landmarks)
    blurred = TF.gaussian_blur(images, [9, 9], sigma=sigma)
    result = images.clone()
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[index, :, y1:y2, x1:x2] = blurred[index, :, y1:y2, x1:x2]
    return result


def condition_name(mode: str, region: str, sigma: float) -> str:
    return f"{mode}_{region}_sigma_{int(sigma):02d}"


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    mode: str | None,
    region_index: int | None,
    sigma: float | None,
    device: torch.device,
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
            batch_landmarks = batch["landmarks"].to(device)
            if mode is not None:
                if region_index is None or sigma is None:
                    raise ValueError("Corruption conditions require region and sigma.")
                if mode in ("rgb_only", "both"):
                    rgb = blur_region(rgb, batch_landmarks, region_index, sigma)
                if mode in ("thermal_only", "both"):
                    thermal = blur_region(thermal, batch_landmarks, region_index, sigma)
            rgb_output = model.forward(
                rgb, batch_landmarks, return_regional_features=True
            )
            thermal_output = model.forward(
                thermal, batch_landmarks, return_regional_features=True
            )
            rgb_features.append(rgb_output["regional_features"].cpu())
            thermal_features.append(thermal_output["regional_features"].cpu())
            landmarks.append(batch_landmarks.cpu())
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


def with_degradation(
    clean: dict[str, object], corrupted: dict[str, object]
) -> dict[str, object]:
    clean_rgb = clean["rgb_regional_features"].float()
    clean_thermal = clean["thermal_regional_features"].float()
    corrupted_rgb = corrupted["rgb_regional_features"].float()
    corrupted_thermal = corrupted["thermal_regional_features"].float()
    clean_distance = 1.0 - F.cosine_similarity(clean_rgb, clean_thermal, dim=-1)
    corrupted_distance = 1.0 - F.cosine_similarity(
        corrupted_rgb, corrupted_thermal, dim=-1
    )
    return {
        "clean_rgb_regional_features": clean_rgb,
        "clean_thermal_regional_features": clean_thermal,
        "corrupted_rgb_regional_features": corrupted_rgb,
        "corrupted_thermal_regional_features": corrupted_thermal,
        "clean_matching_distances": clean_distance,
        "corrupted_matching_distances": corrupted_distance,
        "delta_matching_distances": corrupted_distance - clean_distance,
        "rgb_feature_changes": 1.0
        - F.cosine_similarity(clean_rgb, corrupted_rgb, dim=-1),
        "thermal_feature_changes": 1.0
        - F.cosine_similarity(clean_thermal, corrupted_thermal, dim=-1),
        "landmarks_112": clean["landmarks_112"],
        "pair_key": clean["pair_key"],
        "subject_id": clean["subject_id"],
        "split": "train",
    }


def validate_artifact(path: Path, expected_count: int) -> None:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    tensor_names = (
        "clean_rgb_regional_features",
        "clean_thermal_regional_features",
        "corrupted_rgb_regional_features",
        "corrupted_thermal_regional_features",
    )
    tensors = [artifact[name] for name in tensor_names]
    if any(tuple(tensor.shape) != (expected_count, 4, 100) for tensor in tensors):
        raise AssertionError(f"Invalid feature shape in {path}.")
    for name in (
        "clean_matching_distances",
        "corrupted_matching_distances",
        "delta_matching_distances",
        "rgb_feature_changes",
        "thermal_feature_changes",
    ):
        if tuple(artifact[name].shape) != (expected_count, 4):
            raise AssertionError(f"Invalid target shape for {name} in {path}.")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise AssertionError(f"Non-finite feature tensor in {path}.")
    if not all(
        torch.isfinite(artifact[name]).all()
        for name in (
            "clean_matching_distances",
            "corrupted_matching_distances",
            "delta_matching_distances",
            "rgb_feature_changes",
            "thermal_feature_changes",
        )
    ):
        raise AssertionError(f"Non-finite target tensor in {path}.")
    recomputed_clean = 1.0 - F.cosine_similarity(
        artifact["clean_rgb_regional_features"],
        artifact["clean_thermal_regional_features"],
        dim=-1,
    )
    recomputed_corrupted = 1.0 - F.cosine_similarity(
        artifact["corrupted_rgb_regional_features"],
        artifact["corrupted_thermal_regional_features"],
        dim=-1,
    )
    if not torch.allclose(
        artifact["clean_matching_distances"], recomputed_clean, atol=1e-7, rtol=1e-6
    ):
        raise AssertionError(f"Clean distance mismatch in {path}.")
    if not torch.allclose(
        artifact["corrupted_matching_distances"],
        recomputed_corrupted,
        atol=1e-7,
        rtol=1e-6,
    ):
        raise AssertionError(f"Corrupted distance mismatch in {path}.")
    if not torch.allclose(
        artifact["delta_matching_distances"],
        artifact["corrupted_matching_distances"]
        - artifact["clean_matching_distances"],
        atol=1e-7,
        rtol=1e-6,
    ):
        raise AssertionError(f"Degradation mismatch in {path}.")
    if len(artifact["pair_key"]) != expected_count:
        raise AssertionError(f"Pair-key count mismatch in {path}.")
    if artifact["subject_id"].shape != (expected_count,):
        raise AssertionError(f"Subject-ID shape mismatch in {path}.")
    if artifact["landmarks_112"].shape != (expected_count, 54, 2):
        raise AssertionError(f"Landmark shape mismatch in {path}.")


def summary_rows(
    path: Path, mode: str, region: str, sigma: float, expected_count: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    delta = artifact["delta_matching_distances"].float()
    rgb_change = artifact["rgb_feature_changes"].float()
    thermal_change = artifact["thermal_feature_changes"].float()
    rows = []
    for index, name in enumerate(REGIONS):
        for measure, values in (
            ("delta_matching_distance", delta[:, index]),
            ("rgb_feature_change", rgb_change[:, index]),
            ("thermal_feature_change", thermal_change[:, index]),
        ):
            rows.append(
                {
                    "mode": mode,
                    "corrupted_region": region,
                    "sigma": sigma,
                    "observed_region": name,
                    "measure": measure,
                    "count": expected_count,
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "min": float(values.min()),
                    "q05": float(torch.quantile(values, 0.05)),
                    "median": float(torch.quantile(values, 0.50)),
                    "q95": float(torch.quantile(values, 0.95)),
                    "max": float(values.max()),
                    "negative_fraction": float((values < 0).float().mean()),
                    "near_zero_fraction": float((values.abs() < 1e-5).float().mean()),
                }
            )
    subject_rows = []
    subjects = artifact["subject_id"]
    for subject in torch.unique(subjects).tolist():
        mask = subjects == subject
        for index, name in enumerate(REGIONS):
            values = delta[mask, index]
            subject_rows.append(
                {
                    "mode": mode,
                    "corrupted_region": region,
                    "sigma": sigma,
                    "observed_region": name,
                    "subject_id": int(subject),
                    "count": int(values.numel()),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(unbiased=False)),
                    "negative_fraction": float((values < 0).float().mean()),
                }
            )
    return rows, subject_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E3-E2 clean/corrupted features.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash_before = sha256_file(args.checkpoint)
    model = load_model(args.checkpoint, device)
    parameters_before = parameter_snapshot(model)
    dataset = SFTL54PairDataset(split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    clean_path = args.output / "clean_features.pt"
    started = time.perf_counter()
    clean = extract_features(model, loader, None, None, None, device)
    torch.save(clean, clean_path)
    summary = []
    subject_summary = []
    conditions = []
    for mode in MODES:
        for region_index, region in enumerate(REGIONS):
            for sigma in SIGMAS:
                name = condition_name(mode, region, sigma)
                condition_path = args.output / f"{name}.pt"
                corrupted = extract_features(
                    model, loader, mode, region_index, sigma, device
                )
                artifact = with_degradation(clean, corrupted)
                torch.save(artifact, condition_path)
                validate_artifact(condition_path, len(dataset))
                rows, subject_rows = summary_rows(
                    condition_path, mode, region, sigma, len(dataset)
                )
                summary.extend(rows)
                subject_summary.extend(subject_rows)
                conditions.append(
                    {
                        "name": name,
                        "mode": mode,
                        "corrupted_region": region,
                        "sigma": sigma,
                        "artifact": condition_path.name,
                        "bytes": condition_path.stat().st_size,
                    }
                )
    parameters_after = parameter_snapshot(model)
    if any(
        not torch.equal(parameters_before[name], parameters_after[name])
        for name in parameters_before
    ):
        raise AssertionError("Model parameters changed during E3-E2 extraction.")
    checkpoint_hash_after = sha256_file(args.checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("Checkpoint changed during E3-E2 extraction.")
    pd.DataFrame(summary).to_csv(args.output / "e3_e2_target_summary.csv", index=False)
    pd.DataFrame(subject_summary).to_csv(
        args.output / "e3_e2_subject_target_summary.csv", index=False
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash_before,
        "split": "train",
        "sample_count": len(dataset),
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "regions": list(REGIONS),
        "sigmas": list(SIGMAS),
        "modes": list(MODES),
        "clean_artifact": clean_path.name,
        "conditions": conditions,
        "feature_shape": [len(dataset), 4, 100],
        "distance_shape": [len(dataset), 4],
        "distance_definition": "1 - cosine_similarity(rgb_regional_features, thermal_regional_features, dim=-1)",
        "delta_definition": "corrupted_matching_distances - clean_matching_distances",
        "rgb_feature_change_definition": "1 - cosine_similarity(clean_rgb, corrupted_rgb, dim=-1)",
        "thermal_feature_change_definition": "1 - cosine_similarity(clean_thermal, corrupted_thermal, dim=-1)",
        "negative_delta_preserved": True,
        "clean_images_modified": False,
        "existing_frozen_artifacts_modified": False,
        "model_parameters_unchanged": True,
        "checkpoint_unchanged": True,
        "deterministic_blur_note": "Gaussian blur is deterministic for fixed inputs; repeated stochastic draws were not fabricated.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "e3_e2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report = [
        "# E3-E2 clean and corrupted regional feature construction",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Checkpoint SHA-256: `{checkpoint_hash_before}`",
        f"- Split: `train` ({len(dataset)} samples)",
        "- Corruption modes: RGB-only, thermal-only, and both-modality",
        "- Regions: left eye, right eye, nose, mouth",
        "- Blur strengths: sigma 3, 7, 15",
        "- Features: clean and corrupted regional tensors with shape `[N, 4, 100]`",
        "- Negative degradation values were preserved without clipping or rectification.",
        "",
        "## Target definitions",
        "",
        "- `d_clean = 1 - cosine(rgb_clean, thermal_clean)`",
        "- `d_corrupted = 1 - cosine(rgb_corrupted, thermal_corrupted)`",
        "- `delta_d = d_corrupted - d_clean`",
        "- RGB change: `1 - cosine(rgb_clean, rgb_corrupted)`",
        "- Thermal change: `1 - cosine(thermal_clean, thermal_corrupted)`",
        "",
        "## Repeatability",
        "",
        "The implemented Gaussian blur is deterministic for fixed images, landmarks, "
        "region, and sigma. Therefore repeated identical draws would not measure "
        "stochastic corruption variability and were not fabricated. Future stochastic "
        "corruptions require repeated seeded draws and sign-agreement analysis.",
        "",
        "## Validation",
        "",
        "- All condition artifacts were reloaded and validated.",
        "- Feature, distance, degradation, landmark, pair-key, and subject-ID shapes passed.",
        "- Recomputed clean and corrupted distances matched saved values.",
        "- Recomputed degradation matched `d_corrupted - d_clean`.",
        "- All saved values were finite.",
        "- Model parameters were unchanged.",
        "- The original checkpoint was unchanged.",
        "- Existing E3-D0 frozen artifacts were not overwritten.",
        "",
        "## Files",
        "",
        f"- `clean_features.pt`",
        f"- `{len(conditions)}` corruption condition artifacts",
        "- `e3_e2_target_summary.csv`",
        "- `e3_e2_subject_target_summary.csv`",
        "- `e3_e2_metadata.json`",
        "- `e3_e2_report.md`",
    ]
    (args.output / "e3_e2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
