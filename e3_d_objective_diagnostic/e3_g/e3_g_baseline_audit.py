from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e3_g_evaluation import (
    area_under_risk_coverage,
    calibration_metrics,
    fit_logistic_calibration,
    fit_temperature,
    risk_coverage,
    roc_auc,
    sigmoid,
    tar_at_far,
)
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface

OUTPUT = ROOT / "e3_d_objective_diagnostic" / "e3_g"
FEATURE_ROOT = OUTPUT.parent / "frozen_features"
CHECKPOINT = ROOT / "Experiment_3_Uncertainty_best.pt"
REGIONS = ("left_eye", "right_eye", "nose", "mouth")
SPLITS = ("train", "val", "test")
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(split: str) -> dict[str, object]:
    return torch.load(
        FEATURE_ROOT / f"{split}_features.pt",
        map_location="cpu",
        weights_only=False,
    )


def impostor_indices(subject_ids: np.ndarray) -> np.ndarray:
    groups: dict[int, list[int]] = {}
    for index, subject in enumerate(subject_ids.tolist()):
        groups.setdefault(int(subject), []).append(index)
    subjects = sorted(groups)
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required for impostor construction.")
    result = np.empty(len(subject_ids), dtype=int)
    for position, subject in enumerate(subjects):
        source = groups[subject]
        target = groups[subjects[(position + 1) % len(subjects)]]
        for within, index in enumerate(source):
            result[index] = target[within % len(target)]
    if np.any(subject_ids == subject_ids[result]):
        raise AssertionError("Impostor construction produced a same-subject pair.")
    return result


def extract_global_embeddings(split: str, checkpoint_hash: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = SFTL54PairDataset(split=split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface()))
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rgb_values, thermal_values = [], []
    with torch.no_grad():
        for batch in loader:
            rgb_values.append(F.normalize(model(batch["rgb"]), dim=1).cpu())
            thermal_values.append(F.normalize(model(batch["thermal"]), dim=1).cpu())
    if sha256_file(CHECKPOINT) != checkpoint_hash:
        raise AssertionError("Checkpoint changed during global extraction.")
    return torch.cat(rgb_values), torch.cat(thermal_values)


def make_pairs(split: str, rgb_global: torch.Tensor, thermal_global: torch.Tensor) -> pd.DataFrame:
    artifact = load_features(split)
    subject_ids = artifact["subject_id"].numpy()
    keys = np.asarray(artifact["pair_key"], dtype=str)
    impostors = impostor_indices(subject_ids)
    rows: list[dict[str, object]] = []
    regional_rgb = artifact["rgb_regional_features"]
    regional_thermal = artifact["thermal_regional_features"]
    for label, thermal_index in ((1, np.arange(len(subject_ids))), (0, impostors)):
        regional_similarity = torch.einsum(
            "nrd,nrd->nr", regional_rgb, regional_thermal[thermal_index]
        ).numpy()
        global_similarity = (rgb_global @ thermal_global[thermal_index].T).diagonal().numpy()
        for index in range(len(subject_ids)):
            row: dict[str, object] = {
                "split": split,
                "pair_index": len(rows),
                "label": label,
                "rgb_index": index,
                "thermal_index": int(thermal_index[index]),
                "rgb_subject_id": int(subject_ids[index]),
                "thermal_subject_id": int(subject_ids[thermal_index[index]]),
                "rgb_pair_key": keys[index],
                "thermal_pair_key": keys[thermal_index[index]],
                "global_similarity": float(global_similarity[index]),
            }
            for region_index, region in enumerate(REGIONS):
                row[f"{region}_similarity"] = float(regional_similarity[index, region_index])
                row[f"{region}_distance"] = float(1.0 - regional_similarity[index, region_index])
            rows.append(row)
    return pd.DataFrame(rows)


def score_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, split_frame in frame.groupby("split", sort=False):
        for score_name in ["global_similarity", *[f"{region}_similarity" for region in REGIONS]]:
            values = split_frame[score_name].to_numpy()
            labels = split_frame["label"].to_numpy()
            rows.append(
                {
                    "split": split,
                    "score": score_name,
                    "pair_count": len(split_frame),
                    "genuine_count": int((labels == 1).sum()),
                    "impostor_count": int((labels == 0).sum()),
                    "genuine_mean": float(values[labels == 1].mean()),
                    "genuine_std": float(values[labels == 1].std(ddof=0)),
                    "impostor_mean": float(values[labels == 0].mean()),
                    "impostor_std": float(values[labels == 0].std(ddof=0)),
                    "roc_auc": roc_auc(labels, values),
                    "tar_at_far_001": tar_at_far(labels, values, 0.01),
                    "tar_at_far_010": tar_at_far(labels, values, 0.10),
                }
            )
    return rows


def calibration_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train = frame[frame["split"] == "train"]
    for score_name in ["global_similarity", *[f"{region}_similarity" for region in REGIONS]]:
        train_scores = train[score_name].to_numpy()
        train_labels = train["label"].to_numpy()
        log_scale, log_offset = fit_logistic_calibration(train_scores, train_labels)
        temperature = fit_temperature(train_scores, train_labels)
        for split, split_frame in frame.groupby("split", sort=False):
            scores = split_frame[score_name].to_numpy()
            labels = split_frame["label"].to_numpy()
            methods = {
                "raw_minmax": sigmoid((scores - train_scores.mean()) / max(train_scores.std(), 1e-12)),
                "logistic_train": sigmoid(log_scale * scores + log_offset),
                "temperature_train": sigmoid(scores / temperature),
            }
            for method, probabilities in methods.items():
                metrics = calibration_metrics(labels, probabilities)
                rows.append(
                    {
                        "split": split,
                        "score": score_name,
                        "method": method,
                        **metrics,
                        "roc_auc": roc_auc(labels, scores),
                    }
                )
    return rows


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(CHECKPOINT)
    pair_frames = []
    for split in SPLITS:
        rgb_global, thermal_global = extract_global_embeddings(split, checkpoint_hash)
        pair_frames.append(make_pairs(split, rgb_global, thermal_global))
    pairs = pd.concat(pair_frames, ignore_index=True)
    split_subject_sets = {
        split: set(group["rgb_subject_id"]) | set(group["thermal_subject_id"])
        for split, group in pairs.groupby("split", sort=False)
    }
    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            if split_subject_sets[left_split] & split_subject_sets[right_split]:
                raise AssertionError(
                    f"Subject overlap between {left_split} and {right_split}."
                )
    pairs.to_csv(OUTPUT / "e3_g_pair_scores.csv", index=False)
    pd.DataFrame(score_summary(pairs)).to_csv(
        OUTPUT / "e3_g_regional_score_statistics.csv", index=False
    )
    pd.DataFrame(calibration_summary(pairs)).to_csv(
        OUTPUT / "e3_g_calibration_statistics.csv", index=False
    )
    protocol_rows = []
    for split, group in pairs.groupby("split", sort=False):
        subjects = sorted(set(group["rgb_subject_id"]) | set(group["thermal_subject_id"]))
        protocol_rows.append(
            {
                "split": split,
                "seed": SEED,
                "genuine_pairs": int((group["label"] == 1).sum()),
                "impostor_pairs": int((group["label"] == 0).sum()),
                "total_pairs": len(group),
                "impostor_to_genuine_ratio": float((group["label"] == 0).sum() / (group["label"] == 1).sum()),
                "rgb_subject_count": group["rgb_subject_id"].nunique(),
                "thermal_subject_count": group["thermal_subject_id"].nunique(),
                "union_subject_count": len(subjects),
                "same_subject_genuine_check": bool(
                    (group.loc[group["label"] == 1, "rgb_subject_id"]
                    == group.loc[group["label"] == 1, "thermal_subject_id"]).all()
                ),
                "different_subject_impostor_check": bool(
                    (group.loc[group["label"] == 0, "rgb_subject_id"]
                    != group.loc[group["label"] == 0, "thermal_subject_id"]).all()
                ),
            }
        )
    pd.DataFrame(protocol_rows).to_csv(OUTPUT / "e3_g_pair_statistics.csv", index=False)
    baseline_rows = []
    for split, group in pairs.groupby("split", sort=False):
        for score_name in ["global_similarity", *[f"{region}_similarity" for region in REGIONS]]:
            labels = group["label"].to_numpy()
            confidence = sigmoid(
                (group[score_name].to_numpy() - pairs.loc[pairs["split"] == "train", score_name].mean())
                / max(pairs.loc[pairs["split"] == "train", score_name].std(), 1e-12)
            )
            curve = risk_coverage(labels, confidence)
            baseline_rows.append(
                {
                    "split": split,
                    "score": score_name,
                    "area_under_risk_coverage": area_under_risk_coverage(curve),
                    "risk_at_full_coverage": curve[-1]["risk"],
                    "risk_at_50_percent_coverage": curve[max(0, len(curve) // 2 - 1)]["risk"],
                }
            )
    pd.DataFrame(baseline_rows).to_csv(OUTPUT / "e3_g_selective_baseline_statistics.csv", index=False)
    metadata = {
        "seed": SEED,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "feature_root": str(FEATURE_ROOT),
        "pair_protocol": "all same-row genuine pairs plus one deterministic cyclic different-subject impostor per RGB row",
        "model_training": False,
        "confidence_training": False,
        "global_embedding_extraction": "frozen checkpoint, torch.no_grad, CPU, shuffle=False",
        "test_threshold_tuning": False,
        "regions": REGIONS,
    }
    (OUTPUT / "e3_g_reproducibility.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
