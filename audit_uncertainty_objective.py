from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).parent


def objective(log_uncertainty: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    return torch.exp(-log_uncertainty) * distance + log_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the E3 heteroscedastic objective.")
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=ROOT / "uncertainty_calibration_report.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "uncertainty_objective_audit.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "uncertainty_objective_audit.md",
    )
    args = parser.parse_args()

    distances = torch.tensor(
        [0.0001, 0.0005, 0.0010, 0.001367, 0.0020, 0.0050, 0.01, 0.05, 0.1, 0.5, 1.0],
        dtype=torch.float64,
    )
    bounds = (-4.0, 4.0)
    rows = []
    for distance in distances:
        unconstrained_optimum = math.log(float(distance))
        bounded_optimum = min(max(unconstrained_optimum, bounds[0]), bounds[1])
        for log_uncertainty in (-4.0, -2.0, -1.0, 0.0, 1.0, 4.0):
            u = torch.tensor(log_uncertainty, dtype=torch.float64, requires_grad=True)
            value = objective(u, distance)
            gradient = torch.autograd.grad(value, u)[0]
            rows.append(
                {
                    "distance": float(distance),
                    "log_uncertainty": log_uncertainty,
                    "loss": float(value),
                    "gradient": float(gradient),
                    "unconstrained_optimum": unconstrained_optimum,
                    "bounded_optimum": bounded_optimum,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)

    calibration = pd.read_csv(args.calibration_report)
    clean = calibration[calibration["condition"] == "clean"]
    mean_distance = float(clean["matching_distance"].mean())
    mean_combined = float(clean["combined_log_uncertainty"].mean())
    predicted_optimum = math.log(mean_distance)
    bounded_predicted_optimum = min(max(predicted_optimum, bounds[0]), bounds[1])
    gradient_at_mean = 1.0 - math.exp(-mean_combined) * mean_distance

    summary = f"""# E3 uncertainty objective audit

The E3 objective is:

`L(u, d) = exp(-u) * d + u`

where `u` is combined log uncertainty and `d` is regional cosine distance.
Its exact derivative is:

`dL/du = 1 - exp(-u) * d`

Setting the derivative to zero gives the unconstrained optimum:

`u* = log(d)`

With the implementation bounds `[-4, 4]`, the optimum is
`clamp(log(d), -4, 4)`.

## Observed validation regime

- Mean clean regional distance: `{mean_distance:.9f}`
- Mean trained combined log uncertainty: `{mean_combined:.6f}`
- Unconstrained optimum at mean distance: `{predicted_optimum:.6f}`
- Bounded optimum at mean distance: `{bounded_predicted_optimum:.6f}`
- Derivative at the trained mean: `{gradient_at_mean:.6f}`

For the observed distance scale, `log(d)` is below the lower clamp. The
positive derivative at the trained mean therefore pushes optimization toward
the lower bound for every region. This objective does not distinguish regions
unless their distances differ by enough to produce materially different
`log(d)` values above the clamp. It explains the downward uncertainty trend
and does not by itself establish calibrated reliability.

The numerical grid is saved in `{args.output.name}`. No model or checkpoint
was modified.
"""
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
