from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torchvision.ops import roi_align


PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "sftl54_landmark_config.json"
IMAGE_SIZE = 112.0
REGION_NAMES = ("left_eye", "right_eye", "nose", "mouth")


def load_region_indices(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, tuple[int, ...]]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    regions = config.get("regions")
    if not isinstance(regions, Mapping):
        raise ValueError("Landmark configuration must contain a regions mapping.")
    result: dict[str, tuple[int, ...]] = {}
    for name in REGION_NAMES:
        values = regions.get(name, {}).get("landmark_indices")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Missing landmark indices for region {name!r}.")
        result[name] = tuple(int(value) for value in values)
    return result


class RegionalFeatureExtractor(nn.Module):
    """Pool landmark-defined regions from a spatial feature map with ROIAlign."""

    def __init__(
        self,
        feature_channels: int = 100,
        expansion_margin: float = 0.20,
        output_dim: int | None = None,
        config_path: str | Path = CONFIG_PATH,
    ) -> None:
        super().__init__()
        if expansion_margin < 0:
            raise ValueError("expansion_margin must be non-negative.")
        self.feature_channels = feature_channels
        self.expansion_margin = expansion_margin
        self.output_dim = output_dim or feature_channels
        self.region_indices = load_region_indices(config_path)
        if self.output_dim != feature_channels:
            self.projection = nn.Linear(feature_channels, self.output_dim)
        else:
            self.projection = nn.Identity()

    def image_boxes(
        self,
        landmarks: torch.Tensor,
        image_size: float = IMAGE_SIZE,
    ) -> torch.Tensor:
        if landmarks.ndim != 3 or landmarks.shape[1:] != (54, 2):
            raise ValueError(
                f"landmarks must have shape [B, 54, 2], got {tuple(landmarks.shape)}"
            )
        boxes = []
        for region in REGION_NAMES:
            points = landmarks[:, self.region_indices[region], :]
            minimum = points.amin(dim=1)
            maximum = points.amax(dim=1)
            center = (minimum + maximum) / 2.0
            half_size = (maximum - minimum).clamp_min(1.0) * (1.0 + self.expansion_margin) / 2.0
            region_min = (center - half_size).clamp(0.0, image_size - 1.0)
            region_max = (center + half_size).clamp(1.0, image_size)
            region_max = torch.maximum(region_max, region_min + 1.0).clamp_max(image_size)
            boxes.append(torch.cat((region_min, region_max), dim=1))
        return torch.stack(boxes, dim=1)

    def feature_boxes(
        self,
        landmarks: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        boxes = self.image_boxes(landmarks)
        scale = torch.tensor(
            [width / IMAGE_SIZE, height / IMAGE_SIZE],
            dtype=landmarks.dtype,
            device=landmarks.device,
        )
        scaled = boxes * torch.cat((scale, scale))
        scaled[..., [0, 2]] = scaled[..., [0, 2]].clamp(0.0, float(width))
        scaled[..., [1, 3]] = scaled[..., [1, 3]].clamp(0.0, float(height))
        scaled[..., 2] = torch.maximum(scaled[..., 2], scaled[..., 0] + 1e-3)
        scaled[..., 3] = torch.maximum(scaled[..., 3], scaled[..., 1] + 1e-3)
        return scaled

    def forward(
        self,
        features: torch.Tensor,
        landmarks: torch.Tensor,
        return_boxes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.feature_channels:
            raise ValueError(
                f"features must have shape [B, {self.feature_channels}, H, W], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] != landmarks.shape[0]:
            raise ValueError("Feature and landmark batch sizes must match.")
        batch, _, height, width = features.shape
        boxes = self.feature_boxes(landmarks.to(features), height, width)
        batch_indices = torch.arange(batch, device=features.device, dtype=features.dtype)
        batch_indices = batch_indices[:, None].expand(batch, len(REGION_NAMES))
        roi_boxes = torch.cat((batch_indices[..., None], boxes), dim=-1).reshape(-1, 5)
        pooled = roi_align(
            features,
            roi_boxes,
            output_size=(1, 1),
            spatial_scale=1.0,
            aligned=True,
        ).flatten(1).reshape(batch, len(REGION_NAMES), self.feature_channels)
        projected = self.projection(pooled)
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
        if return_boxes:
            return normalized, boxes
        return normalized

    def coverage(self, landmarks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Return continuous feature-map box widths/heights for diagnostics."""
        boxes = self.feature_boxes(landmarks, height, width)
        return torch.stack((boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]), dim=-1)

    def touched_cell_counts(
        self, landmarks: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Return integer grid-cell spans touched by each regional box."""
        boxes = self.feature_boxes(landmarks, height, width)
        starts = boxes.floor().clamp_min(0)
        ends = boxes.ceil()
        ends[..., [0, 2]] = ends[..., [0, 2]].clamp_max(width)
        ends[..., [1, 3]] = ends[..., [1, 3]].clamp_max(height)
        return torch.stack(
            (
                (ends[..., 2] - starts[..., 0]).clamp_min(1),
                (ends[..., 3] - starts[..., 1]).clamp_min(1),
            ),
            dim=-1,
        ).to(torch.long)
