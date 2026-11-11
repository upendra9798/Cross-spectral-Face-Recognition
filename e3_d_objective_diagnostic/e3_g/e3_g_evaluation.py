"""Statistical utilities for E3-G verification confidence evaluation.

This module contains no model-training code. Inputs are pair-level scores and
binary verification outcomes. All thresholds must be selected outside test
evaluation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def clip_probabilities(probabilities: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = clip_probabilities(probabilities)
    return float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    total = max(len(labels), 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(labels[mask].mean())
        confidence = float(probabilities[mask].mean())
        weight = count / total
        ece += weight * abs(accuracy - confidence)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": abs(accuracy - confidence),
            }
        )
    return float(ece), rows


def reliability_diagram(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float]]:
    """Return reliability-diagram bin statistics without rendering a figure."""
    return expected_calibration_error(labels, probabilities, bins=bins)[1]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    positive_ranks = ranks[labels == 1]
    return float((positive_ranks.sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def tar_at_far(labels: np.ndarray, scores: np.ndarray, far: float) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if len(negatives) == 0 or len(positives) == 0:
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - far, method="higher")
    return float(np.mean(positives >= threshold))


def risk_coverage(
    labels: np.ndarray,
    confidences: np.ndarray,
) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    confidences = np.asarray(confidences, dtype=float)
    order = np.argsort(-confidences, kind="mergesort")
    sorted_labels = labels[order]
    rows: list[dict[str, float]] = []
    for count in range(1, len(sorted_labels) + 1):
        retained = sorted_labels[:count]
        rows.append(
            {
                "coverage": count / len(sorted_labels),
                "risk": float(1.0 - retained.mean()),
                "retained_count": count,
            }
        )
    return rows


def area_under_risk_coverage(rows: Iterable[dict[str, float]]) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    coverage = np.asarray([row["coverage"] for row in rows], dtype=float)
    risk = np.asarray([row["risk"] for row in rows], dtype=float)
    return float(np.trapezoid(risk, coverage) if hasattr(np, "trapezoid") else np.trapz(risk, coverage))


def fit_logistic_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> tuple[float, float]:
    """Fit p(y=1|score)=sigmoid(scale*score+offset) by deterministic GD."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    scale, offset = 1.0, 0.0
    for _ in range(steps):
        probabilities = sigmoid(scale * scores + offset)
        residual = probabilities - labels
        grad_scale = float(np.mean(residual * scores) + l2 * scale)
        grad_offset = float(np.mean(residual))
        scale -= learning_rate * grad_scale
        offset -= learning_rate * grad_offset
    return float(scale), float(offset)


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    steps: int = 2000,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> float:
    """Fit a positive temperature for logits while retaining no intercept."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    log_temperature = 0.0
    for _ in range(steps):
        temperature = float(np.exp(log_temperature))
        logits = scores / temperature
        probabilities = sigmoid(logits)
        residual = probabilities - labels
        gradient_temperature = float(np.mean(residual * (-scores / temperature)))
        gradient_log_temperature = gradient_temperature * temperature + l2 * log_temperature
        log_temperature -= learning_rate * gradient_log_temperature
    return float(np.exp(log_temperature))


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    ece, _ = expected_calibration_error(labels, probabilities)
    return {
        "brier_score": brier_score(labels, probabilities),
        "negative_log_likelihood": negative_log_likelihood(labels, probabilities),
        "expected_calibration_error": ece,
    }
