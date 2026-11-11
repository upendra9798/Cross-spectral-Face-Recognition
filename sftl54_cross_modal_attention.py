from __future__ import annotations

import torch
from torch import nn

from sftl54_transformer_adapter import TransformerAdapter


class BidirectionalCrossModalAttention(nn.Module):
    """Bidirectional RGB/thermal token exchange for paired training samples."""

    def __init__(self, channels: int = 100, heads: int = 4) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(channels)
        self.thermal_norm = nn.LayerNorm(channels)
        self.rgb_attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.thermal_attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.rgb_feed_forward = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.thermal_feed_forward = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        thermal_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_query = self.rgb_norm(rgb_tokens)
        thermal_query = self.thermal_norm(thermal_tokens)
        rgb_update, _ = self.rgb_attention(
            rgb_query, thermal_query, thermal_query, need_weights=False
        )
        thermal_update, _ = self.thermal_attention(
            thermal_query, rgb_query, rgb_query, need_weights=False
        )
        rgb_tokens = rgb_tokens + rgb_update
        thermal_tokens = thermal_tokens + thermal_update
        return (
            rgb_tokens + self.rgb_feed_forward(rgb_tokens),
            thermal_tokens + self.thermal_feed_forward(thermal_tokens),
        )


class EdgeFaceCrossModalAttention(nn.Module):
    """EdgeFace-XS with paired-training cross-modal token interaction."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = TransformerAdapter()
        self.cross_modal = BidirectionalCrossModalAttention()

    def _features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.backbone.model
        x = model.stem(images)
        x = model.stages[0](x)
        x = model.stages[1](x)
        stage2 = model.stages[2](x)
        x = model.stages[3](stage2)
        x = model.norm_pre(x)
        return stage2, model.head(x)

    def _encoded_tokens(self, stage2: torch.Tensor) -> torch.Tensor:
        tokens = stage2.flatten(2).transpose(1, 2)
        tokens = tokens + self.adapter.positional_embedding
        return self.adapter.encoder(tokens)

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.adapter.projection(self.adapter.norm(tokens).mean(dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Single-modality path used for inference and RGB distillation."""
        stage2, backbone_embedding = self._features(images)
        return backbone_embedding + self.adapter(stage2)

    def forward_paired(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend paired samples during training only."""
        rgb_stage2, rgb_backbone = self._features(rgb)
        thermal_stage2, thermal_backbone = self._features(thermal)
        rgb_tokens, thermal_tokens = self.cross_modal(
            self._encoded_tokens(rgb_stage2),
            self._encoded_tokens(thermal_stage2),
        )
        return (
            rgb_backbone + self._project_tokens(rgb_tokens),
            thermal_backbone + self._project_tokens(thermal_tokens),
        )


def configure_cross_modal_student(
    model: EdgeFaceCrossModalAttention,
    include_stage0: bool = True,
) -> list[str]:
    """Freeze EdgeFace and train Transformer/cross-modal adaptation modules."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        is_attention = name.startswith(("adapter.", "cross_modal."))
        is_layer_norm = "norm" in name.lower()
        is_stem = name.startswith("backbone.model.stem.")
        is_stage0 = name.startswith("backbone.model.stages.0.")
        if is_attention or is_layer_norm or is_stem or (include_stage0 and is_stage0):
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names
