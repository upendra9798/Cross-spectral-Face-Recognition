from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface


ROOT = Path(__file__).parent


def check_model(device: torch.device) -> None:
    dataset = SFTL54PairDataset(split="train")
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    rgb = batch["rgb"].to(device)
    thermal = batch["thermal"].to(device)
    landmarks = batch["landmarks"].to(device)
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    model.eval()

    with torch.inference_mode():
        rgb_global = model(rgb)
        thermal_global = model(thermal)
        rgb_output = model(rgb, landmarks, return_regional_features=True)
        thermal_output = model(thermal, landmarks, return_regional_features=True)
        rgb_regions, rgb_boxes = model.regional_extractor(
            torch.randn(2, 100, 7, 7, device=device), landmarks, return_boxes=True
        )

    assert rgb_global.shape == (2, 512)
    assert thermal_global.shape == (2, 512)
    assert rgb_output["global_embedding"].shape == (2, 512)
    assert thermal_output["global_embedding"].shape == (2, 512)
    assert rgb_output["regional_features"].shape == (2, 4, 100)
    assert thermal_output["regional_features"].shape == (2, 4, 100)
    assert torch.isfinite(rgb_output["regional_features"]).all()
    assert torch.isfinite(thermal_output["regional_features"]).all()
    assert torch.isfinite(rgb_boxes).all()
    assert (rgb_boxes[..., 2:] > rgb_boxes[..., :2]).all()
    touched_cells = model.regional_extractor.touched_cell_counts(landmarks, 7, 7)
    assert (touched_cells >= 1).all()
    assert torch.allclose(
        rgb_output["regional_features"].norm(dim=-1),
        torch.ones(2, 4, device=device),
        atol=1e-5,
    )
    print(f"{device}: shapes, finite values, boxes, and normalization passed")

    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(rgb, landmarks, return_regional_features=True)
    loss = output["global_embedding"].sum() + output["regional_features"].sum()
    loss.backward()
    adapter_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith("adapter.") and parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in adapter_gradients)
    print(f"{device}: regional backward gradient passed")


def check_checkpoint() -> None:
    checkpoint_path = ROOT / "Experiment_1_Transformer_Adapter.pt"
    if not checkpoint_path.exists():
        print("Checkpoint compatibility skipped: checkpoint not found")
        return
    model = EdgeFaceTransformerAdapter(load_pretrained_edgeface())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print("Existing Experiment 1 checkpoint loaded successfully")


if __name__ == "__main__":
    check_model(torch.device("cpu"))
    if torch.cuda.is_available():
        check_model(torch.device("cuda"))
    check_checkpoint()
