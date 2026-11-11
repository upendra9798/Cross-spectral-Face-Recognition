from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F


EPSILON = 1e-8


class RankingDetails(NamedTuple):
    loss: torch.Tensor
    valid_pairs: int
    total_pairs: int


def _validate(uncertainty: torch.Tensor, distance: torch.Tensor) -> None:
    if uncertainty.ndim != 2 or distance.ndim != 2:
        raise ValueError("uncertainty and distance must have shape [B, R].")
    if uncertainty.shape != distance.shape:
        raise ValueError("uncertainty and distance shapes must match.")
    if not torch.isfinite(distance).all():
        raise ValueError("distance contains non-finite values.")


def original_heteroscedastic(
    uncertainty: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """Return mean(exp(-u) * d + u), the unchanged E3 reference objective."""
    _validate(uncertainty, distance)
    return (torch.exp(-uncertainty) * distance + uncertainty).mean()


def normalized_heteroscedastic_batch(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize distance by a detached batch mean and return loss, scale."""
    _validate(uncertainty, distance)
    scale = (distance.mean().detach() + epsilon).clamp_min(epsilon)
    normalized_distance = distance / scale
    loss = (torch.exp(-uncertainty) * normalized_distance + uncertainty).mean()
    return loss, scale


def normalized_heteroscedastic_fixed(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    training_scale: float | torch.Tensor,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize distance by a detached training-only fixed scale."""
    _validate(uncertainty, distance)
    scale = torch.as_tensor(training_scale, dtype=distance.dtype, device=distance.device)
    scale = scale.detach().clamp_min(epsilon)
    normalized_distance = distance / scale
    loss = (torch.exp(-uncertainty) * normalized_distance + uncertainty).mean()
    return loss, scale


def pairwise_ranking(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    threshold: float,
    margin: float = 0.0,
) -> RankingDetails:
    """Rank uncertainty in the same direction as regional matching distance."""
    _validate(uncertainty, distance)
    region_count = uncertainty.shape[1]
    losses = []
    valid_pairs = 0
    total_pairs = uncertainty.shape[0] * (region_count * (region_count - 1) // 2)
    for left in range(region_count):
        for right in range(left + 1, region_count):
            delta_distance = distance[:, left] - distance[:, right]
            mask = delta_distance.abs() > threshold
            if mask.any():
                sign = delta_distance[mask].sign()
                delta_uncertainty = uncertainty[mask, left] - uncertainty[mask, right]
                losses.append(F.softplus(margin - sign * delta_uncertainty))
                valid_pairs += int(mask.sum())
    if not losses:
        loss = uncertainty.sum() * 0.0
    else:
        loss = torch.cat(losses).mean()
    return RankingDetails(loss, valid_pairs, total_pairs)


def combined_normalized_ranking(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    training_scale: float | torch.Tensor,
    threshold: float,
    ranking_weight: float = 1.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, RankingDetails, torch.Tensor]:
    """Combine fixed-scale normalized heteroscedastic and ranking losses."""
    normalized_loss, scale = normalized_heteroscedastic_fixed(
        uncertainty, distance, training_scale
    )
    ranking = pairwise_ranking(uncertainty, distance, threshold, margin)
    return normalized_loss + ranking_weight * ranking.loss, ranking, scale


def corruption_supervised(
    clean_uncertainty: torch.Tensor,
    corrupted_uncertainty: torch.Tensor,
    clean_distance: torch.Tensor,
    corrupted_distance: torch.Tensor,
    target_mean: float | torch.Tensor | None = None,
    target_std: float | torch.Tensor | None = None,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match uncertainty change to measured clean-to-corrupted distance degradation."""
    _validate(clean_uncertainty, clean_distance)
    _validate(corrupted_uncertainty, corrupted_distance)
    if clean_uncertainty.shape != corrupted_uncertainty.shape:
        raise ValueError("Clean and corrupted uncertainty shapes must match.")
    degradation = (corrupted_distance - clean_distance).detach()
    mean = (
        degradation.mean().detach()
        if target_mean is None
        else torch.as_tensor(target_mean, device=degradation.device, dtype=degradation.dtype)
    )
    std = (
        degradation.std(unbiased=False).detach()
        if target_std is None
        else torch.as_tensor(target_std, device=degradation.device, dtype=degradation.dtype)
    )
    target = (degradation - mean) / (std.clamp_min(epsilon) + epsilon)
    prediction = corrupted_uncertainty - clean_uncertainty
    return F.huber_loss(prediction, target), target


def calibration_regularization(
    uncertainty: torch.Tensor,
    distance: torch.Tensor,
    epsilon: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sigmoid uncertainty to detached per-sample distance min-max targets."""
    _validate(uncertainty, distance)
    minimum = distance.amin(dim=1, keepdim=True).detach()
    maximum = distance.amax(dim=1, keepdim=True).detach()
    denominator = (maximum - minimum).clamp_min(epsilon)
    target = ((distance.detach() - minimum) / denominator).clamp(0.0, 1.0)
    return F.huber_loss(torch.sigmoid(uncertainty), target), target
