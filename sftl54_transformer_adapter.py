from __future__ import annotations

import torch
from torch import nn


class TransformerAdapter(nn.Module):
    """A small self-attention adapter for EdgeFace-XS Stage 2 features."""

    def __init__(
        self,
        channels: int = 100,
        token_count: int = 49,
        heads: int = 4,
        layers: int = 1,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.positional_embedding = nn.Parameter(torch.zeros(1, token_count, channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(channels)
        self.projection = nn.Linear(channels, embedding_dim)
        # Preserve the pretrained representation at initialization.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        positions = self.positional_embedding
        if positions.shape[1] != height * width:
            positions = positions.transpose(1, 2)
            positions = torch.nn.functional.interpolate(
                positions,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + positions)
        return self.projection(self.norm(tokens).mean(dim=1))


class EdgeFaceTransformerAdapter(nn.Module):
    """Extend EdgeFace-XS without replacing its convolutional backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = TransformerAdapter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone.model
        x = model.stem(x)
        x = model.stages[0](x)
        x = model.stages[1](x)
        stage2 = model.stages[2](x)

        x = model.stages[3](stage2)
        x = model.norm_pre(x)
        backbone_embedding = model.head(x)
        return backbone_embedding + self.adapter(stage2)


def configure_transformer_student(
    model: EdgeFaceTransformerAdapter,
    include_stage0: bool = True,
) -> list[str]:
    """Freeze EdgeFace and train the adapter plus xEdgeFace modules."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        is_adapter = name.startswith("adapter.")
        is_layer_norm = "norm" in name.lower()
        is_stem = name.startswith("backbone.model.stem.")
        is_stage0 = name.startswith("backbone.model.stages.0.")
        if is_adapter or is_layer_norm or is_stem or (include_stage0 and is_stage0):
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names
