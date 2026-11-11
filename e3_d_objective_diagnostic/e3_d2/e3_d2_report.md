# E3-D2 controlled uncertainty-head-only training

## Protocol

All variants trained only the RGB and thermal uncertainty heads using identical:

- Frozen `[B,4,100]` regional feature tensors.
- Initial head parameters.
- Batch ordering (`shuffle=False`).
- Adam optimizer.
- Learning rate `0.001`.
- Batch size `64`.
- Seed `42`.
- Three epochs.

Training constants were fixed from training data only:

- Fixed distance scale: `0.00135366382334`
- Ranking thresholds:
  - p25: `0.000070199370`
  - p50: `0.000161975622`
  - p75: `0.000305652618`

No corruption targets were used for training. Validation blur was evaluation-only.

## Variants

- `original`: original E3 heteroscedastic objective.
- `normalized_fixed`: fixed-scale normalized heteroscedastic objective.
- `ranking_p25`, `ranking_p50`, `ranking_p75`: ranking objective with threshold sensitivity.
- `combined_p25`, `combined_p50`, `combined_p75`: normalized fixed-scale plus ranking.
- `calibration`: calibration regularization objective.

## Main validation results

| Variant | Train loss | Val loss | Combined mean | Combined std | Reliability std | Pearson | Spearman | Ranking accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | -4.351926 | -5.259784 | -5.648867 | 0.012002 | 0.001313 | 0.137053 | 0.132201 | 0.293333 |
| normalized_fixed | 1.000774 | 1.012552 | 0.006015 | 0.001404 | 0.000237 | -0.044767 | -0.058878 | 0.666667 |
| ranking_p25 | 0.682609 | 0.678255 | -1.425819 | 0.057930 | 0.010964 | 0.026112 | 0.035629 | 0.756190 |
| ranking_p50 | 0.680066 | 0.673255 | -1.424150 | 0.059638 | 0.011215 | 0.029641 | 0.040185 | 0.756190 |
| ranking_p75 | 0.676875 | 0.665617 | -1.413123 | 0.059230 | 0.011108 | 0.032510 | 0.042256 | 0.754286 |
| combined_p25 | 1.693580 | 1.705234 | 0.006218 | 0.002109 | 0.000405 | -0.014422 | -0.026278 | 0.702857 |
| combined_p50 | 1.693496 | 1.705075 | 0.006239 | 0.002222 | 0.000431 | -0.011338 | -0.022526 | 0.702857 |
| combined_p75 | 1.693331 | 1.704718 | 0.006271 | 0.002404 | 0.000472 | -0.006797 | -0.016936 | 0.710476 |
| calibration | 0.081424 | 0.083459 | -0.092901 | 0.003433 | 0.000654 | -0.028734 | -0.045122 | 0.714286 |

All variants had zero clamp hits. The ranking variants had a validation
zero-gradient rate of `0.193182`; the other variants had zero-gradient rate
`0.0` under their respective validation objectives. GPU memory was unavailable
because the run used CPU.

## Corruption response

Mean combined uncertainty change across all validation blur conditions:

| Variant | Mean uncertainty change | Mean distance change |
|---|---:|---:|
| original | -0.001996 | 0.000130 |
| normalized_fixed | 0.000114 | 0.000130 |
| ranking_p25 | 0.005618 | 0.000130 |
| ranking_p50 | 0.005828 | 0.000130 |
| ranking_p75 | 0.005647 | 0.000130 |
| combined_p25 | 0.000182 | 0.000130 |
| combined_p50 | 0.000193 | 0.000130 |
| combined_p75 | 0.000209 | 0.000130 |
| calibration | 0.000302 | 0.000130 |

Ranking variants produced the clearest positive corruption response. The
combined variants produced only small positive changes and retained negative
validation correlations. Responses were not uniformly region-specific:
combined variants decreased uncertainty for the nose while matching distance
increased.

## Conclusion

The head-only experiment demonstrates that the ranking objective can create
regional variation and a positive corruption response under frozen features.
However, its Pearson and Spearman correlations with clean matching distance
remain close to zero. Therefore, higher uncertainty variance and stronger blur
response are not sufficient evidence of calibration.

The primary combined objective is numerically stable and respects the intended
scale correction, but it did not improve clean uncertainty/error correlation in
this small run. The ranking-only objective is the better candidate for a
carefully controlled follow-up, with calibration and held-out corruption
generalization still required.

No backbone, Transformer Adapter, regional extractor, original E3 checkpoint,
or reliability-weighted alignment was modified.

## Artifacts

- `e3_d2_train.py`
- `e3_d2_evaluate.py`
- `e3_d2_results.csv`
- `e3_d2_region_results.csv`
- `e3_d2_corruption_results.csv`
- `e3_d2_corruption_summary.csv`
- `e3_d2_corruption_behavior.csv`
- `e3_d2_training_history.csv`
- `training_records.json`
- `e3_d2_checkpoints/`