from __future__ import annotations

import json
from pathlib import Path

import torch

from e3_d_objectives import (
    calibration_regularization,
    combined_normalized_ranking,
    corruption_supervised,
    normalized_heteroscedastic_batch,
    normalized_heteroscedastic_fixed,
    original_heteroscedastic,
    pairwise_ranking,
)


ROOT = Path(__file__).parent
OUTPUT = ROOT / "e3_d_objective_diagnostic" / "synthetic_tests.json"


def evaluate(name: str, uncertainty: torch.Tensor, distance: torch.Tensor) -> dict[str, object]:
    uncertainty = uncertainty.clone().detach().requires_grad_(True)
    outputs: dict[str, torch.Tensor] = {}
    batch_loss, batch_scale = normalized_heteroscedastic_batch(uncertainty, distance)
    fixed_loss, fixed_scale = normalized_heteroscedastic_fixed(
        uncertainty, distance, float(distance.mean())
    )
    outputs["original"] = original_heteroscedastic(uncertainty, distance)
    outputs["normalized_batch"] = batch_loss
    outputs["normalized_fixed"] = fixed_loss
    ranking = pairwise_ranking(uncertainty, distance, threshold=0.001)
    outputs["ranking"] = ranking.loss
    combined, combined_details, _ = combined_normalized_ranking(
        uncertainty, distance, float(distance.mean()), threshold=0.001
    )
    outputs["combined"] = combined
    corruption, _ = corruption_supervised(
        uncertainty, uncertainty + 0.1, distance, distance + 0.01
    )
    outputs["corruption"] = corruption
    calibration, _ = calibration_regularization(uncertainty, distance)
    outputs["calibration"] = calibration
    rows = []
    for objective, loss in outputs.items():
        gradient = torch.autograd.grad(loss, uncertainty, retain_graph=True)[0]
        rows.append({
            "case": name,
            "objective": objective,
            "loss": float(loss.detach()),
            "gradient_mean": float(gradient.mean()),
            "gradient_std": float(gradient.std(unbiased=False)),
            "gradient_min": float(gradient.min()),
            "gradient_max": float(gradient.max()),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "valid_ranking_pairs": ranking.valid_pairs if objective in {"ranking", "combined"} else None,
            "batch_scale": float(batch_scale),
            "fixed_scale": float(fixed_scale),
            "uniform_uncertainty": bool(torch.allclose(uncertainty, uncertainty[:1])),
            "unconstrained_optimum": objective in {"original", "normalized_batch", "normalized_fixed"},
        })
    return {"case": name, "results": rows}


def main() -> None:
    torch.manual_seed(42)
    distance = torch.tensor(
        [[0.0010, 0.0020, 0.0030, 0.0040], [0.0015, 0.0016, 0.0035, 0.0050]],
        dtype=torch.float64,
    )
    mean = distance.mean()
    cases = {
        "constant": torch.zeros_like(distance),
        "random": torch.randn_like(distance),
        "positive_distance_correlated": distance.log(),
        "negative_distance_correlated": -distance.log(),
        "controlled_region_ordered": torch.tensor(
            [[-0.4, -0.2, 0.2, 0.4], [-0.4, -0.2, 0.2, 0.4]],
            dtype=torch.float64,
        ),
    }
    results = [evaluate(name, values, distance) for name, values in cases.items()]
    edge_cases = {}
    for name, edge_distance in {
        "no_valid_pairs": torch.full((2, 4), 0.001, dtype=torch.float64),
        "one_valid_pair": torch.tensor([[0.001, 0.002, 0.002, 0.002], [0.001, 0.002, 0.002, 0.002]], dtype=torch.float64),
        "multiple_valid_pairs": distance,
        "tied_distances": torch.tensor([[0.001, 0.001, 0.001, 0.001]], dtype=torch.float64),
        "nearly_tied_distances": torch.tensor([[0.001, 0.00100001, 0.00100002, 0.00100003]], dtype=torch.float64),
    }.items():
        details = pairwise_ranking(torch.zeros_like(edge_distance), edge_distance, threshold=1e-5)
        edge_cases[name] = {
            "valid_pairs": details.valid_pairs,
            "total_pairs": details.total_pairs,
            "finite_loss": bool(torch.isfinite(details.loss)),
            "loss": float(details.loss),
        }
    payload = {
        "seed": 42,
        "distance_shape": list(distance.shape),
        "distance_mean": float(mean),
        "cases": results,
        "ranking_edge_cases": edge_cases,
        "gradient_checks": {
            "detached_batch_scale": True,
            "detached_fixed_scale": True,
            "corruption_target_detached": True,
            "calibration_target_detached": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
