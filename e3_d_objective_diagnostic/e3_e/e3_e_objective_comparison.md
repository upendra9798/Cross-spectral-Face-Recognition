# E3-E candidate supervision comparison

| Approach | Formulation | Required targets | Gradient behavior | Scale identifiability | Collapse/noise risk | Compute |
|---|---|---|---|---|---|---|
| Bounded pairwise ranking | `softplus(m - s * (tanh(u_l/tau)-tanh(u_r/tau)))` | Relative regional ordering from clean distances | Bounded differences; can saturate at `tanh` limits | No absolute origin; bounded spread only | Less runaway spread, but noisy pair labels remain | Low |
| Probabilistic pairwise ranking | `-log sigmoid(s * (u_l-u_r)/tau)` | Relative ordering | Temperature controls gradient scale; separable pairs still push toward probability 1 | No absolute origin; no finite optimum without an anchor | Pair noise and scale drift remain | Low |
| Degradation supervision | Robust regression from uncertainty change to normalized measured `delta_d` | Clean/corrupted paired regional features and modality-specific perturbations | Direct absolute change gradient; Huber/quantile loss limits outliers | Yes, if train-only normalization and target definition are fixed | Incorrect or unstable corruption targets can teach artifacts | Medium/high |
| Ranking + degradation | Weighted bounded/probabilistic ranking plus degradation regression | Both relative clean distances and paired corruption targets | Ranking supplies ordering; degradation supplies absolute change | Potentially identifiable through degradation term | Weight imbalance and target noise; must monitor each component | Medium/high |
| Region-wise calibration | Per-region bounded regression or monotonic calibration from measured distance/degradation | Region-specific clean or degradation targets | Direct regional gradients; can overfit regions with low sample support | Yes only with a fixed target scale | Region imbalance, subject leakage, and noisy region labels | Medium |

## Evidence from E3-D0 through E3-D4

The current ranking term is demonstrably useful for ordering: validation
ranking accuracy increased to `0.773333`. However, validation combined
Spearman was only `0.057142`, subject bootstrap intervals included zero, and
uncertainty standard deviation increased substantially. This is direct
evidence of scale instability, not evidence that ranking is useless.

The frozen training distances contain nonzero regional variation:

* regional standard deviations: `0.0005366` to `0.0006000`;
* mean absolute pairwise regional differences: approximately
  `0.0002078` to `0.0002495`;
* the train p50 ranking threshold is `0.000161975622`.

These values support a measurable ordinal signal, while the existing
experiments do not establish an absolute reliability target. The available
artifacts do not contain paired corrupted feature tensors, so degradation
supervision cannot yet be judged empirically.

## Recommendation

The primary future strategy should be **measured degradation supervision with
a bounded/probabilistic ranking auxiliary term**:

1. generate paired clean and corrupted RGB-only and thermal-only regional
   features;
2. normalize degradation targets using training data only;
3. use a robust bounded regression term for modality-specific uncertainty
   changes;
4. retain a lower-weight bounded or temperature-controlled ranking term;
5. evaluate per region, per subject, and on held-out corruption conditions.

This is a research recommendation, not a claim of validated superiority. It
addresses the known ranking scale defect while preserving the ordering signal
that E3-D4 actually demonstrated.
