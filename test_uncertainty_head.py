from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface
from uncertainty_head import uncertainty_matching_loss


ROOT = Path(__file__).parent


def main() -> None:
    batch = next(iter(DataLoader(
        SFTL54PairDataset(split="train"),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )))
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface()))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model.forward_with_uncertainty(
        batch["rgb"], batch["thermal"], batch["landmarks"]
    )
    assert output["rgb_uncertainty"].shape == (2, 4)
    assert output["thermal_uncertainty"].shape == (2, 4)
    assert output["regional_reliability"].shape == (2, 4)
    assert output["rgb_regional_features"].shape == (2, 4, 100)
    assert output["thermal_regional_features"].shape == (2, 4, 100)
    assert output["global_rgb_embedding"].shape == (2, 512)
    assert output["global_thermal_embedding"].shape == (2, 512)
    for value in output.values():
        assert torch.isfinite(value).all()
    assert torch.allclose(
        output["regional_reliability"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    loss = uncertainty_matching_loss(
        output["rgb_regional_features"],
        output["thermal_regional_features"],
        output["combined_uncertainty"],
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "uncertainty_head" in name
    )
    optimizer.step()
    assert model(batch["rgb"]).shape == (2, 512)
    checkpoint = ROOT / "Experiment_1_Transformer_Adapter.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        allowed = {
            "rgb_uncertainty_head.network.0.weight",
            "rgb_uncertainty_head.network.0.bias",
            "rgb_uncertainty_head.network.2.weight",
            "rgb_uncertainty_head.network.2.bias",
            "thermal_uncertainty_head.network.0.weight",
            "thermal_uncertainty_head.network.0.bias",
            "thermal_uncertainty_head.network.2.weight",
            "thermal_uncertainty_head.network.2.bias",
        }
        assert set(missing) == allowed and not unexpected
    print("uncertainty shapes, stability, reliability, gradients, and compatibility passed")


if __name__ == "__main__":
    main()
