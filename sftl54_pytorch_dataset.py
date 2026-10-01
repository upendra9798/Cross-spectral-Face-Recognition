from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).parent
DEFAULT_MANIFEST = PROJECT_ROOT / "sftl54_processed" / "manifest.csv"


class SFTL54PairDataset(Dataset):
    """Dataset of paired 112x112 RGB and thermal face crops."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        split: str | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path

        self.records = pd.read_csv(self.manifest_path)
        if split is not None:
            valid_splits = {"train", "val", "test"}
            if split not in valid_splits:
                raise ValueError(f"split must be one of {sorted(valid_splits)}, got {split!r}")
            self.records = self.records[self.records["split"] == split].reset_index(drop=True)

        required_columns = {"rgb_path", "thermal_path", "subject_id"}
        missing_columns = required_columns - set(self.records.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records.iloc[index]
        rgb_path = self._resolve_path(record["rgb_path"])
        thermal_path = self._resolve_path(record["thermal_path"])

        # EdgeFace expects three-channel input for both modalities.
        rgb_image = Image.open(rgb_path).convert("RGB")
        thermal_image = Image.open(thermal_path).convert("L").convert("RGB")

        return {
            "rgb": self.transform(rgb_image),
            "thermal": self.transform(thermal_image),
            "label": torch.tensor(int(record["subject_id"]), dtype=torch.long),
            "subject_id": int(record["subject_id"]),
            "pair_key": record["pair_key"],
        }


if __name__ == "__main__":
    train_dataset = SFTL54PairDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))

    print("Dataset size:", len(train_dataset))
    print("RGB batch:", tuple(batch["rgb"].shape))
    print("Thermal batch:", tuple(batch["thermal"].shape))
    print("Label batch:", tuple(batch["label"].shape))
    print("Subject IDs:", batch["subject_id"].tolist())
    print("Pair keys:", batch["pair_key"])
