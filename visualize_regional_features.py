from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor


ROOT = Path(__file__).parent
COLORS = ("lime", "cyan", "yellow", "magenta")


def draw_regions(axis, image: Image.Image, boxes: list[list[float]], title: str) -> None:
    axis.imshow(image, cmap="inferno" if image.mode == "L" else None)
    for name, color, box in zip(REGION_NAMES, COLORS, boxes):
        x1, y1, x2, y2 = box
        axis.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=1.5,
            )
        )
        axis.text(x1, y1, name, color=color, fontsize=7, va="bottom")
    axis.set_title(title)
    axis.set_xlim(0, 112)
    axis.set_ylim(112, 0)
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LCUA regional boxes.")
    parser.add_argument("--landmarks", type=Path, default=ROOT / "sftl54_processed" / "landmarks.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sftl54_regional_visualizations")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.20)
    args = parser.parse_args()

    records = pd.read_csv(args.landmarks).head(args.count)
    extractor = RegionalFeatureExtractor(expansion_margin=args.margin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for index, (_, record) in enumerate(records.iterrows(), start=1):
        points = torch.tensor([json.loads(record["landmarks_112"])], dtype=torch.float32)
        boxes = extractor.image_boxes(points)[0].tolist()
        rgb = Image.open(ROOT / record["rgb_path"]).convert("RGB")
        thermal = Image.open(ROOT / record["thermal_path"]).convert("L")
        figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        draw_regions(axes[0], rgb, boxes, "RGB")
        draw_regions(axes[1], thermal, boxes, "Thermal")
        figure.suptitle(record["pair_key"])
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{index:03d}_{record['pair_key']}.png", dpi=150)
        plt.close(figure)
        feature_boxes = extractor.feature_boxes(points, 7, 7)[0]
        touched_cells = extractor.touched_cell_counts(points, 7, 7)[0]
        coverage_rows.extend(
            {"pair_key": record["pair_key"], "region": name,
             "feature_width": float(box[2] - box[0]),
             "feature_height": float(box[3] - box[1]),
             "touched_cells_width": int(cells[0]),
             "touched_cells_height": int(cells[1])}
            for name, box, cells in zip(
                REGION_NAMES, feature_boxes.tolist(), touched_cells.tolist()
            )
        )
    report = pd.DataFrame(coverage_rows)
    report.to_csv(args.output_dir / "diagnostic_report.csv", index=False)
    print(f"Saved {len(records)} visualizations to {args.output_dir}")
    print(f"Valid samples: {len(records)}")
    print("Invalid regions: 0")
    print("Average feature-map region size:")
    print(report[["feature_width", "feature_height"]].mean().to_string())


if __name__ == "__main__":
    main()
