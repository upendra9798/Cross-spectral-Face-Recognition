from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "sftl54_landmark_config.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "sftl54_processed" / "manifest.csv"
OUTPUT_SIZE = 112
FACE_MARGIN = 0.35
LANDMARK_GROUPS: tuple[tuple[str, int], ...] = (
    ("chin", 17),
    ("left_eyebrow", 5),
    ("right_eyebrow", 5),
    ("nose_bridge", 4),
    ("nose_tip", 5),
    ("left_eye", 6),
    ("right_eye", 6),
    ("lips", 6),
)


def pair_key_from_filename(filename: str) -> str:
    return Path(os.path.basename(filename)).stem.rsplit("_", 1)[0]


def make_square_box(
    face_box: list[list[float]],
    image_width: int,
    image_height: int,
    margin: float = FACE_MARGIN,
) -> tuple[int, int, int, int]:
    x1, y1 = face_box[0]
    x2, y2 = face_box[1]
    size = max(x2 - x1, y2 - y1) * (1.0 + margin)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = max(0, int(round(center_x - size / 2.0)))
    crop_y1 = max(0, int(round(center_y - size / 2.0)))
    crop_x2 = min(image_width, int(round(center_x + size / 2.0)))
    crop_y2 = min(image_height, int(round(center_y + size / 2.0)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Invalid crop box {(crop_x1, crop_y1, crop_x2, crop_y2)}")
    return crop_x1, crop_y1, crop_x2, crop_y2


def transform_point(
    point: Iterable[float],
    crop_box: tuple[int, int, int, int],
    output_size: int = OUTPUT_SIZE,
) -> list[float]:
    x, y = point
    x1, y1, x2, y2 = crop_box
    return [
        (float(x) - x1) * output_size / (x2 - x1),
        (float(y) - y1) * output_size / (y2 - y1),
    ]


def _point_group(record: dict[str, Any], key: str) -> list[list[float]]:
    value = record.get(key)
    if value is None and key.startswith("lip_"):
        value = record.get(key.replace("lip_", "lip-"))
    if value is None or len(value) == 0:
        raise ValueError(f"Missing landmark group {key!r} for {record.get('image')!r}")
    return value


def flatten_landmarks(record: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for group, expected_count in LANDMARK_GROUPS:
        source_key = group
        if group == "lips":
            source_keys = [f"lip_p{i}" for i in range(1, 7)]
            group_points = [_point_group(record, key)[0] for key in source_keys]
        else:
            group_points = _point_group(record, source_key)
        if len(group_points) != expected_count:
            raise ValueError(
                f"Expected {expected_count} points in {source_key!r}, "
                f"got {len(group_points)}"
            )
        points.extend(group_points)
    if len(points) != 54:
        raise AssertionError(f"Expected 54 landmarks, got {len(points)}")
    return points


def transform_record(record: dict[str, Any], output_size: int = OUTPUT_SIZE) -> dict[str, Any]:
    crop_box = make_square_box(record["face"], record["width"], record["height"])
    points = flatten_landmarks(record)
    return {
        "image": os.path.basename(record["image"]),
        "pair_key": pair_key_from_filename(record["image"]),
        "source_size": [record["width"], record["height"]],
        "crop_box": list(crop_box),
        "landmarks": [
            transform_point(point, crop_box, output_size) for point in points
        ],
    }


def load_landmark_records(
    archive_path: str | Path,
    split: str,
    required_pair_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    labels_path = f"sftl/iron/{split}/labels.jsonl"
    with zipfile.ZipFile(archive_path) as archive:
        records = [
            json.loads(line)
            for line in archive.read(labels_path).decode("utf-8").splitlines()
            if line.strip()
        ]
    transformed: dict[str, dict[str, Any]] = {}
    for record in records:
        pair_key = pair_key_from_filename(record["image"])
        if required_pair_keys is not None and pair_key not in required_pair_keys:
            continue
        processed = transform_record(record)
        key = processed["pair_key"]
        if key in transformed:
            raise ValueError(f"Duplicate landmark pair key {key!r} in {labels_path}")
        transformed[key] = processed
    return transformed


def build_processed_landmarks(
    archive_path: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for split, split_manifest in manifest.groupby("split", sort=False):
        required_pair_keys = set(split_manifest["pair_key"])
        records = load_landmark_records(archive_path, split, required_pair_keys)
        for row in split_manifest.to_dict("records"):
            landmark = records.get(row["pair_key"])
            if landmark is None:
                raise KeyError(
                    f"No landmark annotation for {row['pair_key']!r} in split {split!r}"
                )
            rows.append(
                {
                    **row,
                    "landmark_image": landmark["image"],
                    "source_size": json.dumps(landmark["source_size"]),
                    "crop_box": json.dumps(landmark["crop_box"]),
                    "landmarks_112": json.dumps(landmark["landmarks"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform SFTL54 landmarks to 112x112 crops.")
    parser.add_argument("--zip", required=True, type=Path, help="Official SFTL54 sftl.zip")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "sftl54_processed" / "landmarks.csv",
    )
    args = parser.parse_args()
    result = build_processed_landmarks(args.zip, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} landmark records to {args.output}")


if __name__ == "__main__":
    main()
