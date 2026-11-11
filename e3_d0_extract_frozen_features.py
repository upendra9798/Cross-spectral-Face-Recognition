from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from regional_feature_extractor import RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = ROOT / "Experiment_3_Uncertainty_best.pt"
DEFAULT_OUTPUT = ROOT / "e3_d_objective_diagnostic" / "frozen_features"
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def load_model(checkpoint: Path, device: torch.device) -> EdgeFaceTransformerAdapter:
    teacher = load_pretrained_edgeface()
    model = EdgeFaceTransformerAdapter(copy.deepcopy(teacher)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_split(
    model: EdgeFaceTransformerAdapter,
    split: str,
    batch_size: int,
    device: torch.device,
    output_path: Path,
) -> dict[str, object]:
    dataset = SFTL54PairDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    extractor = model.regional_extractor
    rgb_features = []
    thermal_features = []
    distances = []
    landmarks = []
    boxes = []
    valid_regions = []
    pair_keys: list[str] = []
    subject_ids = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmark_batch = batch["landmarks"].to(device)
            output = model.forward_with_uncertainty(rgb, thermal, landmark_batch)
            rgb_batch = output["rgb_regional_features"].cpu()
            thermal_batch = output["thermal_regional_features"].cpu()
            distance_batch = 1.0 - F.cosine_similarity(
                rgb_batch, thermal_batch, dim=-1
            )
            box_batch = extractor.image_boxes(landmark_batch).cpu()
            valid_batch = (
                (box_batch[..., 2] > box_batch[..., 0])
                & (box_batch[..., 3] > box_batch[..., 1])
                & torch.isfinite(box_batch).all(dim=-1)
            )
            rgb_features.append(rgb_batch)
            thermal_features.append(thermal_batch)
            distances.append(distance_batch)
            landmarks.append(landmark_batch.cpu())
            boxes.append(box_batch)
            valid_regions.append(valid_batch)
            pair_keys.extend(str(value) for value in batch["pair_key"])
            subject_ids.extend(int(value) for value in batch["subject_id"])
    artifact = {
        "rgb_regional_features": torch.cat(rgb_features),
        "thermal_regional_features": torch.cat(thermal_features),
        "matching_distances": torch.cat(distances),
        "landmarks_112": torch.cat(landmarks),
        "region_boxes_112": torch.cat(boxes),
        "valid_regions": torch.cat(valid_regions),
        "pair_key": pair_keys,
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "split": split,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    elapsed = time.perf_counter() - start
    return {
        "split": split,
        "sample_count": len(dataset),
        "batch_size": batch_size,
        "extraction_seconds": elapsed,
        "artifact": output_path.name,
        "artifact_bytes": output_path.stat().st_size,
        "invalid_region_count": int((~artifact["valid_regions"]).sum()),
        "invalid_sample_count": int((~artifact["valid_regions"].all(dim=1)).sum()),
        "rgb_shape": list(artifact["rgb_regional_features"].shape),
        "thermal_shape": list(artifact["thermal_regional_features"].shape),
        "distance_shape": list(artifact["matching_distances"].shape),
    }


def validate_artifact(path: Path, expected_split: str) -> dict[str, object]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    rgb = artifact["rgb_regional_features"]
    thermal = artifact["thermal_regional_features"]
    saved_distance = artifact["matching_distances"]
    recomputed = 1.0 - F.cosine_similarity(rgb, thermal, dim=-1)
    if artifact["split"] != expected_split:
        raise AssertionError(f"{path}: split metadata mismatch")
    if rgb.ndim != 3 or tuple(rgb.shape[1:]) != (4, 100):
        raise AssertionError(f"{path}: invalid RGB shape {tuple(rgb.shape)}")
    if thermal.shape != rgb.shape or saved_distance.shape != rgb.shape[:2]:
        raise AssertionError(f"{path}: incompatible tensor shapes")
    if not all(torch.isfinite(value).all() for value in (rgb, thermal, saved_distance)):
        raise AssertionError(f"{path}: non-finite tensor value")
    if not torch.allclose(saved_distance, recomputed, atol=1e-7, rtol=1e-6):
        raise AssertionError(f"{path}: saved distances do not match recomputation")
    if len(artifact["pair_key"]) != rgb.shape[0]:
        raise AssertionError(f"{path}: identifier count mismatch")
    if artifact["subject_id"].shape != (rgb.shape[0],):
        raise AssertionError(f"{path}: subject identifier count mismatch")
    if artifact["landmarks_112"].shape != (rgb.shape[0], 54, 2):
        raise AssertionError(f"{path}: landmark shape mismatch")
    if artifact["valid_regions"].shape != (rgb.shape[0], 4):
        raise AssertionError(f"{path}: validity shape mismatch")
    return {
        "split": expected_split,
        "sample_count": int(rgb.shape[0]),
        "distance_max_abs_error": float((saved_distance - recomputed).abs().max()),
        "all_finite": True,
        "normalized_rgb_mean_norm": float(rgb.norm(dim=-1).mean()),
        "normalized_thermal_mean_norm": float(thermal.norm(dim=-1).mean()),
        "invalid_region_count": int((~artifact["valid_regions"]).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-D0 frozen regional feature extraction.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash_before = sha256_file(args.checkpoint)
    model = load_model(args.checkpoint, device)
    parameters_before = parameter_snapshot(model)
    args.output.mkdir(parents=True, exist_ok=True)
    extraction = []
    for split in SPLITS:
        extraction.append(
            extract_split(
                model,
                split,
                args.batch_size,
                device,
                args.output / f"{split}_features.pt",
            )
        )
    parameters_after = parameter_snapshot(model)
    if any(not torch.equal(parameters_before[name], parameters_after[name]) for name in parameters_before):
        raise AssertionError("Model parameters changed during frozen extraction.")
    checkpoint_hash_after = sha256_file(args.checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("Checkpoint changed during extraction.")

    validation = []
    split_subjects: dict[str, set[int]] = {}
    split_pairs: dict[str, set[str]] = {}
    for split in SPLITS:
        path = args.output / f"{split}_features.pt"
        validation.append(validate_artifact(path, split))
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        split_subjects[split] = set(int(value) for value in artifact["subject_id"])
        split_pairs[split] = set(artifact["pair_key"])
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            if split_subjects[left_split] & split_subjects[right_split]:
                raise AssertionError(f"Subject overlap: {left_split}/{right_split}")
            if split_pairs[left_split] & split_pairs[right_split]:
                raise AssertionError(f"Pair overlap: {left_split}/{right_split}")

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash_after,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "model_parameters_unchanged": True,
        "checkpoint_unchanged": True,
        "regional_feature_shape": [4, 100],
        "distance_shape": [4],
        "feature_normalization": "RegionalFeatureExtractor applies L2 normalization over the last dimension.",
        "distance_calculation": "1 - cosine_similarity(rgb_regional_features, thermal_regional_features, dim=-1)",
        "invalid_sample_handling": "Samples are preserved; valid_regions records per-region non-zero finite boxes.",
        "ordering": "DataLoader uses shuffle=False; pair_key order is preserved within each split.",
        "split_subject_overlap_checked": True,
        "split_pair_overlap_checked": True,
        "extraction": extraction,
        "validation": validation,
        "subject_counts": {split: len(values) for split, values in split_subjects.items()},
        "split_counts": {row["split"]: row["sample_count"] for row in extraction},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_lines = [
        "# E3-D0 frozen regional feature extraction",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Checkpoint SHA-256: `{checkpoint_hash_after}`",
        f"- Device: `{device}`",
        f"- Seed: `{args.seed}`",
        f"- Batch size: `{args.batch_size}`",
        "- Model parameters modified: no",
        "- Checkpoint modified: no",
        "- Feature shape per sample: `[4, 100]`",
        "- Distance shape per sample: `[4]`",
        "- Feature normalization: L2 normalization over the feature dimension.",
        "- Distance: `1 - cosine_similarity(rgb, thermal)` per region.",
        "- Invalid samples: preserved and flagged in `valid_regions`; no invalid regions were silently dropped.",
        "",
        "## Split extraction",
        "",
        "| Split | Samples | Subjects | Invalid regions | File | Size (bytes) | Seconds |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in extraction:
        split = row["split"]
        report_lines.append(
            f"| {split} | {row['sample_count']} | {metadata['subject_counts'][split]} | "
            f"{row['invalid_region_count']} | {row['artifact']} | {row['artifact_bytes']} | "
            f"{row['extraction_seconds']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Validation",
            "",
            "All artifacts were reloaded successfully. Shapes, finite values, saved "
            "distances, identifier counts, landmark shapes, region-validity shapes, "
            "subject-disjoint splits, pair-disjoint splits, checkpoint bytes, and "
            "model parameters were verified.",
        ]
    )
    (args.output / "metadata_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
