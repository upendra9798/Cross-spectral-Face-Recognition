from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sftl54_cross_modal_attention import EdgeFaceCrossModalAttention
from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_xedgeface_train import (
    extract_embeddings,
    load_pretrained_edgeface,
    retrieval_metrics,
)


ROOT = Path(__file__).parent


def load_experiment_model(checkpoint_path: Path) -> torch.nn.Module:
    model = EdgeFaceCrossModalAttention(copy.deepcopy(load_pretrained_edgeface()))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Experiment 2.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "Experiment_2_Cross_Modal_Attention.pt",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_experiment_model(checkpoint_path).to(device)
    loader = DataLoader(
        SFTL54PairDataset(split="test"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    rgb, thermal, ids = extract_embeddings(model, loader, device)
    metrics = retrieval_metrics(rgb, thermal, ids, ids)
    print("Checkpoint:", checkpoint_path)
    print("Test pairs:", len(ids))
    print("RGB embeddings:", tuple(rgb.shape))
    print("Thermal embeddings:", tuple(thermal.shape))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
