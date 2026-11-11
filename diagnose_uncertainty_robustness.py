from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from regional_feature_extractor import REGION_NAMES, RegionalFeatureExtractor
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent


def blur_region(images: torch.Tensor, landmarks: torch.Tensor, region: str) -> torch.Tensor:
    extractor = RegionalFeatureExtractor()
    boxes = extractor.image_boxes(landmarks)
    region_index = REGION_NAMES.index(region)
    result = images.clone()
    blurred_images = TF.gaussian_blur(images, kernel_size=[9, 9], sigma=3.0)
    for index, box in enumerate(boxes[:, region_index].tolist()):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        result[index, :, y1:y2, x1:x2] = blurred_images[index, :, y1:y2, x1:x2]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure uncertainty under regional blur.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "uncertainty_robustness.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    loader = DataLoader(
        SFTL54PairDataset(split="val"), batch_size=min(8, args.count), shuffle=False
    )
    rows = []
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            if seen >= args.count:
                break
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            landmarks = batch["landmarks"].to(device)
            baseline = model.forward_with_uncertainty(rgb, thermal, landmarks)
            for region in REGION_NAMES:
                blurred_rgb = blur_region(rgb, landmarks, region)
                blurred_thermal = blur_region(thermal, landmarks, region)
                blurred = model.forward_with_uncertainty(
                    blurred_rgb, blurred_thermal, landmarks
                )
                for index in range(min(rgb.shape[0], args.count - seen)):
                    rows.append({
                        "pair_key": batch["pair_key"][index],
                        "region": region,
                        "baseline_rgb": float(baseline["rgb_uncertainty"][index, REGION_NAMES.index(region)]),
                        "blurred_rgb": float(blurred["rgb_uncertainty"][index, REGION_NAMES.index(region)]),
                        "delta_rgb": float(blurred["rgb_uncertainty"][index, REGION_NAMES.index(region)] - baseline["rgb_uncertainty"][index, REGION_NAMES.index(region)]),
                        "baseline_thermal": float(baseline["thermal_uncertainty"][index, REGION_NAMES.index(region)]),
                        "blurred_thermal": float(blurred["thermal_uncertainty"][index, REGION_NAMES.index(region)]),
                        "delta_thermal": float(blurred["thermal_uncertainty"][index, REGION_NAMES.index(region)] - baseline["thermal_uncertainty"][index, REGION_NAMES.index(region)]),
                    })
            seen += min(rgb.shape[0], args.count - seen)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"saved {len(rows)} robustness rows to {args.output}")


if __name__ == "__main__":
    main()
