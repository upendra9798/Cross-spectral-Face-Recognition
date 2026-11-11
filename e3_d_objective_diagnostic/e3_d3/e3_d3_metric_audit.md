# E3-D3 metric audit

## Current E3-D2 definitions

The current evaluator is `e3_d2_evaluate.py`.

### Pearson correlation

`evaluate_clean()` flattens the complete `[N, 4]` uncertainty tensor and the
complete `[N, 4]` matching-distance tensor before calling `torch.corrcoef`.
Therefore the reported Pearson values are global sample-region correlations.
They are not per-region, per-sample, or correlations of aggregated means.
RGB, thermal, and combined uncertainty are each available separately, but the
main D2 result table reports combined uncertainty.

### Spearman correlation

Spearman is computed by applying `argsort().argsort()` to the flattened
uncertainty and distance tensors and then calling the same Pearson helper.
It is also a global sample-region statistic. Ties are not explicitly handled
with average ranks; ordinal ranks are assigned by the double argsort.

### Ranking accuracy

`ranking_accuracy()` operates within each sample. It compares all six unordered
region pairs, retains pairs with `abs(distance_i-distance_j) > threshold`, and
counts whether the uncertainty difference has the same sign as the distance
difference. Counts are pooled across samples and region pairs. Exact ties and
pairs below threshold are excluded. The D2 table uses combined uncertainty.

### Reliability standard deviation

Reliability is computed as `softmax(-combined_uncertainty, dim=1)` and the
reported standard deviation is over all samples and all four regions. It is not
a per-region or per-sample average.

### Corruption uncertainty change

The D2 corruption evaluator corrupts one target region at a time and records
the uncertainty and distance change for that target region only. It does not
record unrelated-region responses. The CSV contains RGB, thermal, and combined
changes, with validation samples and sigma 3, 7, and 15.

## Metric comparability

The global correlations, pooled within-sample ranking accuracy, global
reliability standard deviation, and target-region corruption changes are
different aggregation levels. They must not be interpreted as interchangeable
measurements.
