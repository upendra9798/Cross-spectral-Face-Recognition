from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from sftl54_landmark_processor import build_processed_landmarks


PROJECT_ROOT = Path(__file__).parent
DEFAULT_MANIFEST = PROJECT_ROOT / "sftl54_processed" / "manifest.csv"
REGIONS = {
    "left_eye": [36, 37, 38, 39, 40, 41],
    "right_eye": [42, 43, 44, 45, 46, 47],
    "nose": [27, 28, 29, 30, 31, 32, 33, 34, 35],
    "mouth": [48, 49, 50, 51, 52, 53],
}
COLORS = {
    "left_eye": "lime",
    "right_eye": "cyan",
    "nose": "yellow",
    "mouth": "magenta",
}


def draw_landmarks(axis, image: Image.Image, points: list[list[float]]) -> None:
    axis.imshow(image, cmap="inferno" if image.mode == "L" else None)
    for region, indices in REGIONS.items():
        xs = [points[index][0] for index in indices]
        ys = [points[index][1] for index in indices]
        axis.scatter(xs, ys, s=12, c=COLORS[region], label=region)
    axis.set_xlim(0, 112)
    axis.set_ylim(112, 0)
    axis.axis("off")


def save_pair_visualization(row: pd.Series, output_path: Path) -> None:
    points = json.loads(row["landmarks_112"])
    rgb = Image.open(PROJECT_ROOT / row["rgb_path"]).convert("RGB")
    thermal = Image.open(PROJECT_ROOT / row["thermal_path"]).convert("L")
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    draw_landmarks(axes[0], rgb, points)
    axes[0].set_title("RGB")
    draw_landmarks(axes[1], thermal, points)
    axes[1].set_title("Thermal")
    axes[0].legend(loc="lower left", fontsize=7)
    figure.suptitle(row["pair_key"])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SFTL54 landmark alignment.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "sftl54_landmark_visualizations",
    )
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    rows = build_processed_landmarks(args.zip, args.manifest).head(args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        save_pair_visualization(row, args.output_dir / f"{index:03d}_{row['pair_key']}.png")
    print(f"Saved {len(rows)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
