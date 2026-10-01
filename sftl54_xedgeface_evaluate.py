from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_xedgeface_train import (
    extract_embeddings,
    load_pretrained_edgeface,
    retrieval_metrics,
)

ROOT = Path(__file__).parent


def load_adapted_model(checkpoint_path: Path) -> torch.nn.Module:
    model = load_pretrained_edgeface()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./sftl54_xedgeface_best.pt",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_adapted_model(checkpoint_path).to(device)

    test_loader = DataLoader(
        SFTL54PairDataset(split="test"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    rgb_embeddings, thermal_embeddings, subject_ids = extract_embeddings(
        model,
        test_loader,
        device,
    )
    metrics = retrieval_metrics(
        rgb_embeddings,
        thermal_embeddings,
        subject_ids,
        subject_ids,
    )

    print("Checkpoint:", checkpoint_path)
    print("Test pairs:", len(subject_ids))
    print("RGB embeddings:", tuple(rgb_embeddings.shape))
    print("Thermal embeddings:", tuple(thermal_embeddings.shape))
    print("xEdgeFace test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
