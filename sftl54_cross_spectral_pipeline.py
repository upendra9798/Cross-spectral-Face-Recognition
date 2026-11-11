from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from huggingface_hub import hf_hub_download
from PIL import Image


# This script downloads the SFTL54 cross-spectral face dataset, matches RGB
# images with their thermal counterparts, validates the matches, crops faces,
# and writes a model-ready directory structure plus a CSV manifest.
DATASET_REPO = "issai/SFTL54"
DATASET_FILE = "sftl.zip"


def _parse_pair_key(filename: str) -> str:
    """Remove the final image variant number to obtain the pair identifier."""
    name = os.path.basename(filename) # 111_1_2_1_120_100_1.jpg
    stem = os.path.splitext(name)[0] # 111_1_2_1_120_100_1
    return stem.rsplit("_", 1)[0] #111_1_2_1_120_100
#  111_1_2_1_120_100 That becomes the pair_key.
# So potentially:
# RGB:
# 111_1_2_1_120_100_1.jpg
# Thermal:
# 111_1_2_1_120_100_3.jpg
# both become:
# 111_1_2_1_120_100
# Therefore the code knows:
# RGB ─────────┐
#              ├── pair_key = 111_1_2_1_120_100
# Thermal ─────┘

def _pairing_key_is_valid(filename: str) -> bool:
    """Check the expected filename shape used by the pairing logic."""
    name = os.path.basename(filename)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    return len(parts) >= 6 and parts[0].isdigit() and parts[-1] in {"1", "3"}


def load_metadata(zip_path: str) -> pd.DataFrame:
    # """Read subject-level metadata from the Excel file inside the ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as z:
        df = pd.read_excel(io.BytesIO(z.read("sftl/metadata.xlsx")), header=1)#converts those bytes into something pandas can read.
    df = df.rename(columns={"Unnamed: 0": "subject_id", "Unnamed: 1": "gender"})
    df["subject_id"] = df["subject_id"].astype(int)
    return df[["subject_id", "gender", "Trial_1", "Trial_2"]].copy()


def _candidate_paths_in_split(zf: zipfile.ZipFile, split: str, modality: str) -> List[str]:
    # """Return image paths for one split and modality from the open archive."""
    modality_prefix = {
        "rgb": f"sftl/rgb/{split}/",
        "thermal": f"sftl/iron/{split}/images/",
    }[modality]

    return [
        p for p in zf.namelist()
        if p.startswith(modality_prefix) and p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def build_paired_dataset(zip_path: str | None = None) -> pd.DataFrame:#Find every valid RGB + thermal pair and create a DataFrame describing them.
    """Build one row per complete RGB/thermal pair and print pairing diagnostics."""
    if zip_path is None:
        zip_path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
        #If no ZIP path is provided, download it automatically.

    with zipfile.ZipFile(zip_path, "r") as zf:
        metadata_df = load_metadata(zip_path)
        rows: List[dict] = []

        # Collect every image first so duplicate and incomplete pair keys can be diagnosed.
        for split in ["train", "val", "test"]:
            rgb_paths = _candidate_paths_in_split(zf, split, "rgb")
            thermal_paths = _candidate_paths_in_split(zf, split, "thermal")

            for modality, paths in [("rgb", rgb_paths), ("thermal", thermal_paths)]:
                for path in paths:
                    name = os.path.basename(path)
                    stem = os.path.splitext(name)[0]#111_1_2_1_120_100_1
                    subject_id = int(stem.split("_")[0])#111
                    pair_key = _parse_pair_key(name)
                    valid = _pairing_key_is_valid(name)
                    rows.append(
                        {
                            "split": split,
                            "modality": modality,
                            "subject_id": subject_id,
                            "pair_key": pair_key,
                            "filename": name,
                            "path_in_zip": path,
                            "stem": stem,
                            "pair_key_valid": valid,
                        }
                    )

    df = pd.DataFrame(rows)
    df = df.merge(metadata_df, on="subject_id", how="left")
#     Before:
# subject_id
# pair_key
# RGB path
# thermal path

# After:
# subject_id
# gender
# Trial_1
# Trial_2
# ...

    print("Images by modality:")
    print(df.groupby(["split", "modality"]).size()) #The code prints image counts:
#Something like:
# split  modality
# train  rgb         ... 30
#        thermal     ...10

    keys = ["split", "modality", "pair_key"]
    duplicates = df[df.duplicated(keys, keep=False)]#Detect duplicate pairs
# Suppose there are two RGB images with:
# train + rgb + 111_1_2_1_120_100
# Then that's suspicious.
# The code detects it.

    print("\nDuplicate pairing keys:", len(duplicates))
    print(
        duplicates[["split", "modality", "filename", "pair_key"]].head(20).to_string(index=False)
    )

    counts = df.groupby(["split", "pair_key", "modality"]).size().unstack(fill_value=0)
    for col in ["rgb", "thermal"]:
        if col not in counts:
            counts[col] = 0
# This creates something conceptually like: 
# | split | pair_key | rgb | thermal |
# | ----- | -------- | --: | ------: |
# | train | A        |   1 |       1 |
# | train | B        |   1 |       0 |
# | train | C        |   0 |       1 |
# (counts["rgb"] == 1) & (counts["thermal"] == 1)
# means: Exactly one RGB and exactly one thermal image exist.
# The script also reports:
# RGB without thermal
# Thermal without RGB


    print("\nExactly one RGB and one thermal:")
    print(((counts["rgb"] == 1) & (counts["thermal"] == 1)).sum())

    print("\nRGB without thermal:")
    print(((counts["rgb"] > 0) & (counts["thermal"] == 0)).sum())

    print("\nThermal without RGB:")
    print(((counts["thermal"] > 0) & (counts["rgb"] == 0)).sum())

    invalid_rows = df[~df["pair_key_valid"]]
    if not invalid_rows.empty:
        print("\nInvalid filename keys:")
        print(invalid_rows[["split", "modality", "filename", "pair_key"]].head(20).to_string(index=False))

    # Pivot modality rows into RGB and thermal columns, then keep complete pairs only.
    paired = df.pivot_table(
        index=["split", "subject_id", "pair_key"],
        columns="modality",
        values="path_in_zip",
        aggfunc="first",
    ).reset_index()

    paired = paired.rename(columns={"rgb": "rgb_path", "thermal": "thermal_path"})
    paired["pair_is_complete"] = paired["rgb_path"].notna() & paired["thermal_path"].notna()
    paired = paired[paired["pair_is_complete"]].copy()
    paired = paired.merge(metadata_df, on="subject_id", how="left")
    paired = paired.sort_values(["split", "subject_id", "pair_key"]).reset_index(drop=True)

    final_cols = ["subject_id", "split", "pair_key", "rgb_path", "thermal_path", "gender", "Trial_1", "Trial_2"]
    paired = paired[[c for c in final_cols if c in paired.columns]]
    
    # So one row represents:
    #    ONE PERSON
    #         +
    #   ONE RGB IMAGE
    #         +
    #  ONE THERMAL IMAGE

    print("\nPair completeness summary:")
    print(paired["split"].value_counts().to_string())
    return paired


def verify_subject_disjoint_split(df: pd.DataFrame) -> pd.DataFrame:
    # """Check whether any subject appears in more than one dataset split."""
    split_subjects = (
        df.groupby("split")["subject_id"].apply(lambda s: set(s.unique())).to_dict()
    )
    overlaps = {
        f"{a}_vs_{b}": split_subjects[a] & split_subjects[b]
        for a in split_subjects
        for b in split_subjects
        if a < b
    }

    summary = []
    for split, subjects in split_subjects.items():
        summary.append({"split": split, "n_subjects": len(subjects)})

    summary_df = pd.DataFrame(summary)
    summary_df["subject_disjoint"] = True
    if any(overlaps.values()):
        summary_df["subject_disjoint"] = False

    return summary_df, overlaps


def inspect_pair_landmarks(
    df: pd.DataFrame,
    zip_path: str,
    pair_key: str = "111_1_2_1_120_100",
) -> dict:
    # """Find and print the thermal face annotation for one paired sample."""
    matches = df[df["pair_key"] == pair_key]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paired row for {pair_key!r}, found {len(matches)}.")

    pair = matches.iloc[0]
    labels_path = f"sftl/iron/{pair['split']}/labels.jsonl"
    landmark_record = None

    # Landmark records are stored in one JSONL file per split and reference image names.
    with zipfile.ZipFile(zip_path, "r") as zf:
        thermal_name = os.path.basename(pair["thermal_path"])
        for line in zf.read(labels_path).decode("utf-8").splitlines():
            record = json.loads(line)
            if os.path.basename(record.get("image", "")) == thermal_name:
                landmark_record = record
                break

    if landmark_record is None:
        raise KeyError(f"No landmark record found for {thermal_name!r} in {labels_path!r}.")

    print("RGB path:", pair["rgb_path"])
    print("Thermal path:", pair["thermal_path"])
    print("Landmarks:")
    print(json.dumps(landmark_record, indent=2))
    return landmark_record


def inspect_pair_dimensions(
    df: pd.DataFrame,
    zip_path: str,
    pair_key: str,
) -> None:
    # """Print the dimensions and pixel modes of both images in one pair."""
    matches = df[df["pair_key"] == pair_key]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paired row for {pair_key!r}, found {len(matches)}.")

    pair = matches.iloc[0]
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(pair["rgb_path"]) as rgb_file:
            rgb = Image.open(rgb_file)
            rgb.load()

        with zf.open(pair["thermal_path"]) as thermal_file:
            thermal = Image.open(thermal_file)
            thermal.load()

    print("RGB:")
    print("  size:", rgb.size)
    print("  mode:", rgb.mode)
    print("\nThermal:")
    print("  size:", thermal.size)
    print("  mode:", thermal.mode)


def make_square_box(
    box: list[list[float]],
    image_width: int,
    image_height: int,
    margin: float = 0.35,
) -> tuple[int, int, int, int]:
    """Expand a face box into a square crop, clipped to image boundaries."""
    x1, y1 = box[0]
    x2, y2 = box[1]

    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size = max(width, height) * (1 + margin)

    crop_x1 = max(0, int(round(center_x - size / 2)))#0-prevents coordinates from going negative.
    crop_y1 = max(0, int(round(center_y - size / 2)))
    crop_x2 = min(image_width, int(round(center_x + size / 2)))
    crop_y2 = min(image_height, int(round(center_y + size / 2)))
    return crop_x1, crop_y1, crop_x2, crop_y2

# Example:
# margin = 0.35
# face = 100 × 120
# maximum:
# 120
# with margin:
# 120 × 1.35 = 162
# So instead of tightly cropping the face, it creates a larger square around it.

def crop_face(
    image: Image.Image,
    face_box: list[list[float]],
    size: int = 112,
) -> Image.Image:
    # """Crop a face region and resize it to the model input size."""
    box = make_square_box(face_box, image.width, image.height)
    face = image.crop(box)
    return face.resize((size, size), Image.Resampling.BILINEAR)# resizes it to:112 × 112

# So:
#     Original image
#       ↓
# Face bounding box
#       ↓
# Square crop + 35% margin
#       ↓
# 112 × 112


def test_face_crop(#This function visually verifies that cropping works correctly.
    df: pd.DataFrame,
    zip_path: str,
    pair_key: str,
    landmark_record: dict,
    output_path: str = "sftl54_face_crop_test.png",
) -> None:
    # """Create a visual check showing RGB and thermal crops for one pair."""
    matches = df[df["pair_key"] == pair_key]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paired row for {pair_key!r}, found {len(matches)}.")

    pair = matches.iloc[0]
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(pair["rgb_path"]) as rgb_file:
            rgb = Image.open(rgb_file).convert("RGB")
        with zf.open(pair["thermal_path"]) as thermal_file:
            thermal = Image.open(thermal_file).convert("L")

    rgb_face = crop_face(rgb, landmark_record["face"])
    thermal_face = crop_face(thermal, landmark_record["face"])

    figure, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(rgb_face)
    axes[0].set_title("RGB Face (112x112)")
    axes[1].imshow(thermal_face, cmap="inferno")
    axes[1].set_title("Thermal Face (112x112)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("Saved crop test:", output_path)
    print("RGB crop:", rgb_face.size, rgb_face.mode)
    print("Thermal crop:", thermal_face.size, thermal_face.mode)


def _load_landmark_faces(zf: zipfile.ZipFile, split: str) -> dict[str, list[list[float]]]:
    # """Load face boxes keyed by thermal image filename for one split."""
    labels_path = f"sftl/iron/{split}/labels.jsonl"
    faces = {}
    for line in zf.read(labels_path).decode("utf-8").splitlines():
        record = json.loads(line)
        image_name = os.path.basename(record.get("image", ""))
        if image_name and "face" in record:
            faces[image_name] = record["face"]
    return faces


def preprocess_pairs(
    df: pd.DataFrame,
    zip_path: str,
    output_dir: str = "sftl54_processed",
) -> pd.DataFrame:
    # """Crop all pairs, save them as 112x112 PNGs, and write a manifest CSV."""
    output_root = Path(output_dir)
    manifest_rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            faces = _load_landmark_faces(zf, split)
            rgb_dir = output_root / split / "rgb"
            thermal_dir = output_root / split / "thermal"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            thermal_dir.mkdir(parents=True, exist_ok=True)

            # Use the thermal annotation for both modalities because the paired images
            # represent the same face and share the dataset's coordinate system.
            for _, pair in split_df.iterrows():
                thermal_name = os.path.basename(pair["thermal_path"])
                if thermal_name not in faces:
                    raise KeyError(f"No face annotation found for {thermal_name!r}.")

                with zf.open(pair["rgb_path"]) as rgb_file:
                    rgb = Image.open(rgb_file).convert("RGB")
                    rgb.load()
                with zf.open(pair["thermal_path"]) as thermal_file:
                    thermal = Image.open(thermal_file).convert("L")
                    thermal.load()

                face_box = faces[thermal_name]
                rgb_face = crop_face(rgb, face_box)
                thermal_face = crop_face(thermal, face_box)
                rgb_output = rgb_dir / f"{pair['pair_key']}.png"
                thermal_output = thermal_dir / f"{pair['pair_key']}.png"
                rgb_face.save(rgb_output, format="PNG")
                thermal_face.save(thermal_output, format="PNG")
                manifest_rows.append(
                    {
                        "subject_id": pair["subject_id"],
                        "split": split,
                        "pair_key": pair["pair_key"],
                        "rgb_path": str(rgb_output),
                        "thermal_path": str(thermal_output),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Saved processed pairs:", len(manifest))
    print("Processed manifest:", manifest_path)
    return manifest


def verify_processed_dataset(
    manifest_path: str = "sftl54_processed/manifest.csv",
) -> pd.DataFrame:
    # """Inspect the generated manifest and confirm a sample can be opened."""
    processed = pd.read_csv(manifest_path)

    print("\nProcessed dataset:")
    print(processed.groupby("split").size().to_string())

    print("\nSubjects:")
    print(processed.groupby("split")["subject_id"].nunique().to_string())

    print("\nMissing RGB:")
    print(processed["rgb_path"].isna().sum())

    print("\nMissing Thermal:")
    print(processed["thermal_path"].isna().sum())

    row = processed.iloc[0]
    rgb = Image.open(row["rgb_path"])
    thermal = Image.open(row["thermal_path"])
    print("\nProcessed pair:")
    print("RGB:", rgb.size, rgb.mode)
    print("Thermal:", thermal.size, thermal.mode)
    return processed


def save_pair_visualization(
    df: pd.DataFrame,
    zip_path: str,
    output_path: str = "sftl54_pair_visualization.png",
    n_pairs: int = 8,
) -> None:
    # """Save a contact sheet of RGB and thermal images from the paired dataset."""
    sample = df.sort_values(["split", "subject_id", "pair_key"]).head(n_pairs)
    n_rows = len(sample)
    if n_rows == 0:
        raise ValueError("Cannot visualize an empty paired dataset.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        figure, axes = plt.subplots(n_rows, 2, figsize=(10, 3.2 * n_rows), squeeze=False)
        for row_index, (_, pair) in enumerate(sample.iterrows()):
            with zf.open(pair["rgb_path"]) as rgb_file:
                rgb_image = Image.open(rgb_file).convert("RGB")
            with zf.open(pair["thermal_path"]) as thermal_file:
                thermal_image = Image.open(thermal_file).convert("L")

            axes[row_index, 0].imshow(rgb_image)
            axes[row_index, 1].imshow(thermal_image, cmap="inferno")
            axes[row_index, 0].set_title(f"RGB | {pair['split']} | subject {pair['subject_id']}")
            axes[row_index, 1].set_title(f"Thermal | pair {pair['pair_key']}")
            axes[row_index, 0].axis("off")
            axes[row_index, 1].axis("off")

        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

    print("Saved visualization:", output_path)


if __name__ == "__main__":
    # Run the complete pipeline when this file is executed directly.
    zip_path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    df = build_paired_dataset(zip_path)
    df.to_csv("sftl54_paired_dataset.csv", index=False)
    print("Saved:", len(df), "pairs")
    print("Paired rows:", len(df))
    print(df.head().to_string(index=False))
    print("\nSplit counts:")
    print(df["split"].value_counts().to_string())
    summary_df, overlaps = verify_subject_disjoint_split(df)
    print("\nSubject-disjoint split summary:")
    print(summary_df.to_string(index=False))
    print("\nSplit overlaps:", {k: len(v) for k, v in overlaps.items()})
    save_pair_visualization(df, zip_path)
    inspect_pair_dimensions(
        df,
        zip_path,
        "111_1_2_1_120_100",
    )
    landmarks = inspect_pair_landmarks(df, zip_path, "111_1_2_1_120_100")
    test_face_crop(df, zip_path, "111_1_2_1_120_100", landmarks)
    preprocess_pairs(df, zip_path)
    verify_processed_dataset()
