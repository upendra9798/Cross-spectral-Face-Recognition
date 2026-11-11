# E3-F proposed experiment protocol

## Scope

Investigation-to-implementation bridge only. This protocol does not authorize
training in E3-F.

## Data split

1. Train subjects: fit a small confidence model or estimate calibration
   parameters.
2. Validation subjects: select calibration method and operating threshold
   under a pre-registered rule.
3. Test subjects: evaluate once after all choices are frozen.

No sample-level random split may cross subject boundaries.

## Pair construction

Construct genuine and impostor RGB-thermal comparison episodes with explicit
pair identifiers and subject labels. Keep the original paired sample order
for genuine comparisons. Generate impostor comparisons only between disjoint
subjects, balancing or recording the sampling ratio.

Report pair counts and subject overlap for every split.

## Frozen inputs

Use the existing:

* 7x7 Stage 2 regional features;
* Transformer Adapter outputs;
* landmark-guided four-region extraction;
* global 512-dimensional embeddings;
* clean E3-D0 feature artifacts where applicable.

Do not modify the backbone, Adapter, regional extractor, or checkpoints in
the first experiment.

## Candidate score families

### Baseline

```text
s0_r = -d_rgb_thermal(r)
```

### Calibrated confidence

```text
s_r = h_theta(
  d_rgb_thermal(r),
  ||z_rgb(r)||,
  ||z_thermal(r)||,
  agreement_features(r),
  consistency_features(r)
)
p_r = sigmoid(s_r)
```

Fit `h_theta` with binary cross-entropy or a calibration method on
subject-disjoint outcomes. Keep consistency and agreement features optional
ablations, not required labels.

## Required ablations

1. Distance only.
2. Distance plus feature norms.
3. Distance plus cross-modal agreement.
4. Distance plus transformation consistency.
5. All features.
6. Region-specific versus shared calibration parameters.
7. With and without global embedding features.
8. Genuine-only same-pair evaluation versus genuine/impostor episodes.

## Evaluation

For each region and aggregate:

* ROC-AUC and PR-AUC;
* Brier score;
* expected calibration error with documented bins;
* adaptive calibration error;
* calibration slope and intercept;
* negative log-likelihood;
* risk-coverage curves;
* selective risk at fixed coverage;
* per-subject bootstrap confidence intervals;
* performance by RGB/thermal modality condition;
* cross-region error coupling;
* compute and memory cost.

Do not use increased confidence variance as evidence of calibration.

## Decision rule

Advance only if the calibrated regional score:

1. improves held-out Brier/ECE over the distance-only baseline;
2. improves or preserves verification discrimination;
3. reduces selective risk at a pre-registered coverage;
4. has no region with materially degraded calibration;
5. generalizes across validation and test subjects;
6. remains useful when cross-modal agreement is ablated.

If agreement ablation removes all benefit, describe the result as calibrated
matching confidence rather than independent reliability.
