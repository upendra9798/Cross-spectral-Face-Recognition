# Raw uncertainty and gradient saturation audit

Checkpoint: `Experiment_3_Uncertainty_best.pt`

## Conclusion

**A and D: objective formulation is primary, with the resulting collapse amplified by the feature representation.**

Clean raw uncertainty pooled standard deviation was `0.096335` and clean
reliability standard deviation was `0.000095`. The clean clamp-hit
rate was `0.000000`. Mean targeted-corruption matching-distance
change was `0.000100799`.

The raw outputs, clamped outputs, combined uncertainty, reliability, and
matching distance were captured on the same validation pairs. Raw values are
the direct final MLP outputs before the head's clamp; clamped values use
`[-4, 4]`.

## Raw statistics

```text
modality      mean      std   minimum   maximum       p05       p25       p50       p75       p95
     rgb -0.528059 0.000799 -0.529874 -0.525277 -0.529185 -0.528622 -0.528172 -0.527612 -0.526552
 thermal -0.335463 0.000673 -0.337239 -0.333594 -0.336483 -0.335962 -0.335489 -0.334967 -0.334312
```

## Gradient audit

The gradient audit differentiates the implemented clamped objective with
respect to raw head outputs. A zero gradient indicates that a value is outside
the clamp interval and is blocked by `torch.clamp`; values inside the interval
retain the objective gradient.

```text
modality      loss            objective  gradient_mean  gradient_std  gradient_min  gradient_max  zero_gradient_count  near_lower_clamp_count  near_upper_clamp_count  lower_clamped_count  upper_clamped_count  total_count
     rgb -0.860281 combined_rgb_thermal       0.996759      0.001125      0.991306      0.998934                    0                       0                       0                    0                    0          704
 thermal -0.860281 combined_rgb_thermal       0.996759      0.001125      0.991306      0.998934                    0                       0                       0                    0                    0          704
```

## Interpretation

The existing objective has optimum `u*=log(d)`. The audit confirms whether the
observed raw outputs are actually clamp-saturated and whether clamp-induced
zero gradients explain the lack of learning. If raw values are not at the
clamp while reliability is uniform, clamp saturation is not the main cause;
the objective's common-distance optimum and the low-resolution regional
features are more plausible contributors. No model, objective, checkpoint, or
training run was changed.
