# E3-D1 mathematical objective diagnostics

No model or uncertainty head was trained. All objective tests use synthetic uncertainty tensors and frozen E3-D0 regional distances.

## E3-D0 integrity

- Checkpoint SHA-256: `821c312dc332ecbfed64272b50e867bf72dff2bc341518df284226228c86fa03`
- Split counts: `{'train': 1772, 'val': 176, 'test': 565}`
- Feature shape per sample: `[4, 100]`
- Distance shape per sample: `[4]`
- Feature normalization: L2-normalized regional features.
- Distance: `1 - cosine_similarity(rgb, thermal)`.

## Training-derived constants

- Fixed training distance scale: `0.00135366382334`
- Ranking thresholds: `[0.0, 7.019937038421631e-05, 0.00016197562217712402, 0.0003056526184082031]`
- All thresholds and the fixed scale were estimated from training distances only.
- Validation and test distances were not used to select constants.

## Frozen distance findings

| Split | Mean | Std | Minimum | Maximum |
|---|---:|---:|---:|---:|
| train | 0.001353664 | 0.000584361 | 0.000263453 | 0.008313715 |
| val | 0.001366640 | 0.000474560 | 0.000449777 | 0.003667414 |
| test | 0.001455418 | 0.000656322 | 0.000316739 | 0.006027400 |

Regional and split-specific statistics are in `e3_d1_distance_statistics.csv`.

## Ranking-pair availability

Pair statistics are in `e3_d1_ranking_pair_statistics.csv`. A pair is valid only when the absolute regional distance difference exceeds the training-derived threshold. Ties and nearly tied pairs are excluded.

At the training-derived median threshold `0.000161975622177`, training has 5316 valid of 10632 pairs; validation and test counts are reported without using them to choose the threshold.

## Objective conclusions

- Original heteroscedastic loss has an unconstrained optimum `u=log(d)` and therefore pushes toward low uncertainty for these small distances.
- Normalized heteroscedastic losses remove the absolute scale mismatch, but retain an optimum based on normalized distance and do not add supervision.
- Ranking loss is mathematically valid when enough unequal regional pairs exist; it constrains ordering but not absolute uncertainty calibration.
- The combined objective is mathematically justified as a scale-corrected magnitude term plus a relative-order term, but it is not evidence of calibration.
- Corruption supervision is implemented as an interface only. No corruption targets were fabricated and no corruption training was run.
- Calibration regularization uses detached per-sample min-max distance targets; tied samples produce a stable finite target rather than a gradient path.

## Mathematical test status

Synthetic test source: `e3_d1_synthetic_tests.py`
Synthetic results path: `synthetic_tests.json`
All losses are expected to remain finite. Detached normalization and detached calibration/corruption targets prevent hidden gradients through statistics.
`e3_d1_objective_comparison.csv` records finite-loss/gradient status and confirms that no model training was performed.

## Recommendation for E3-D2

Proceed first with the combined fixed-scale normalized heteroscedastic plus ranking objective only if the ranking-pair table shows sufficient non-tied pairs. Retain normalized-only and ranking-only as ablations. Do not select an objective based on loss decrease alone; require validation uncertainty/error correlation and held-out corruption response.

Diagnostic runtime: `0.024` seconds.