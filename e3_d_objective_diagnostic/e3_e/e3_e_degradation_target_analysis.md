# E3-E degradation-target analysis

## Scope and available evidence

The desired target is:

```text
d_clean       = clean regional RGB-thermal matching distance
d_corrupted   = corrupted regional RGB-thermal matching distance
delta_d       = d_corrupted - d_clean
```

The E3-D0 artifacts contain clean frozen features only:

* train: 1,772 samples;
* validation: 176 samples;
* test: 565 samples;
* RGB and thermal regional tensors: `[N, 4, 100]`;
* matching distances: `[N, 4]`.

No clean/corrupted regional feature-tensor pairs are preserved in the
available E3-D0 artifacts. E3-D2 through E3-D4 corruption CSVs contain
model-level uncertainty and distance changes for live image corruption, but
they do not provide a reusable per-sample pair of corrupted RGB and thermal
regional feature tensors for constructing new training targets. Therefore no
degradation targets are generated in E3-E.

## Required future target construction

For each training sample and selected corruption condition, a future
preprocessing/evaluation pipeline must preserve:

```text
clean_rgb_region, clean_thermal_region
corrupted_rgb_region, corrupted_thermal_region
d_clean[region], d_corrupted[region], delta_d[region]
corruption_type, severity, target_region, pair_key, subject_id
```

The target should be computed from measured feature distances, not from blur
severity. Severity is an experimental condition and can be used for
stratification, but it must not replace `delta_d`.

## Separate modality targets

A single cross-modal distance does not identify whether the degradation came
from RGB, thermal, or both modalities. A future design should therefore retain
modality-specific perturbations:

```text
delta_rgb_target   = distance(corrupted_rgb, clean_thermal)
                     - distance(clean_rgb, clean_thermal)

delta_thermal_target = distance(clean_rgb, corrupted_thermal)
                       - distance(clean_rgb, clean_thermal)
```

For joint corruption, the cross-modal `delta_d` can be retained as an
additional target, but it should not be interpreted as a uniquely RGB or
thermal target.

## Stability checks required before training

For the training split only, future target generation must report:

1. mean, standard deviation, median, and quantiles of `delta_d`;
2. the negative-degradation fraction;
3. the near-zero fraction under a pre-registered tolerance;
4. per-region and per-subject distributions;
5. repeatability across random seeds or repeated corruption draws;
6. correlation between RGB-only and thermal-only target changes;
7. the fraction of samples with target sign disagreement across repeats;
8. feature-norm and cosine-similarity changes.

Negative values are not automatically invalid. They can occur because of
feature noise, landmark/ROI variation, stochastic corruption, or a corruption
that accidentally improves cross-modal agreement. They should be retained for
regression analysis, with robust losses and repeated-condition summaries used
to limit the effect of noisy observations.

## Expected target issues

The clean frozen distances already show regional variation, but that does not
establish that corruption-induced changes are large or stable. A future
experiment must compare the magnitude of `delta_d` with its repeated-draw
noise and with clean sample-to-sample variability. If the target standard
deviation is close to the repeatability noise, degradation supervision will
mostly teach corruption randomness rather than reliability.

## Current conclusion

Degradation supervision is feasible as an interface, but its label validity is
unresolved until paired corrupted feature tensors are generated and audited.
No claim of target quality or superiority is made here.
