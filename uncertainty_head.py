from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RegionalUncertaintyHead(nn.Module):
    """Predict bounded log uncertainty independently for four regional features."""

    def __init__(
        self,
        feature_dim: int = 100,
        hidden_dim: int = 32,
        min_log_uncertainty: float = -4.0,
        max_log_uncertainty: float = 4.0,
    ) -> None:
        super().__init__()
        if min_log_uncertainty >= max_log_uncertainty:
            raise ValueError("min_log_uncertainty must be less than the maximum.")
        self.min_log_uncertainty = min_log_uncertainty
        self.max_log_uncertainty = max_log_uncertainty
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, regional_features: torch.Tensor) -> torch.Tensor:
        if regional_features.ndim != 3:
            raise ValueError(
                f"regional_features must have shape [B, R, D], got {tuple(regional_features.shape)}"
            )
        raw = self.network(regional_features).squeeze(-1)
        return torch.clamp(
            raw,
            min=self.min_log_uncertainty,
            max=self.max_log_uncertainty,
        )


def regional_reliability(
    rgb_log_uncertainty: torch.Tensor,
    thermal_log_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb_log_uncertainty.shape != thermal_log_uncertainty.shape:
        raise ValueError("RGB and thermal uncertainty shapes must match.")
    combined = rgb_log_uncertainty + thermal_log_uncertainty
    weights = F.softmax(-combined, dim=1)
    return combined, weights


def uncertainty_matching_loss(
    rgb_features: torch.Tensor,
    thermal_features: torch.Tensor,
    combined_log_uncertainty: torch.Tensor,
) -> torch.Tensor:
    if rgb_features.shape != thermal_features.shape:
        raise ValueError("RGB and thermal regional feature shapes must match.")
    if combined_log_uncertainty.shape != rgb_features.shape[:2]:
        raise ValueError("Uncertainty must have shape [B, R].")
    distance = 1.0 - F.cosine_similarity(rgb_features, thermal_features, dim=-1)
    weighted = torch.exp(-combined_log_uncertainty) * distance
    return (weighted + combined_log_uncertainty).mean()
