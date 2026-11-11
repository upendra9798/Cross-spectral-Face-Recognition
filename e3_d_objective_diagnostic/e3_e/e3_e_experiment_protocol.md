# E3-E proposed experiment protocol

## Objective

Test whether measured regional degradation can identify modality-specific
reliability while preserving the relative ordering signal from clean matching
distances. This protocol is proposed only; no training is authorized by E3-E
Step 1.

## Data and model controls

* Use the existing 7x7 Stage 2 regional representation.
* Use the existing frozen Transformer Adapter and landmark-guided regions:
  left eye, right eye, nose, and mouth.
* Keep the EdgeFace-XS backbone, Transformer Adapter, regional extractor, and
  cached clean regional features frozen.
* Use the existing subject-disjoint train/validation/test split.
* Train only the RGB and thermal uncertainty heads.
* Keep sample and modality ordering paired.
* Use train-only target normalization and thresholds.
* Reserve validation for checkpoint selection and test for final evaluation.

## Corruption separation

Construct training conditions from a pre-registered set of corruption types,
target regions, and severities. For every training sample, preserve clean and
corrupted RGB-only, thermal-only, and optionally joint-corruption features.

Use different held-out corruption strengths and, where possible, a held-out
corruption type for validation. Do not select corruption strengths or target
transformations from validation/test statistics. If repeated draws are used,
assign the same draw policy to each split before training begins.

## Candidate loss

The recommended candidate is:

```text
L = L_degradation_rgb
  + L_degradation_thermal
  + lambda_rank * L_bounded_rank
```

For modality `q`, predict clean and corrupted uncertainty and regress:

```text
target_q = normalize_train(delta_q)
prediction_q = u_corrupted_q - u_clean_q
L_degradation_q = Huber(prediction_q, target_q)
```

Use a fixed training mean and scale, robustly estimated without validation or
test data. Keep `lambda_rank` pre-registered and report component losses
separately. A probabilistic ranking temperature must also be fixed from
training data or a documented design constant; it must not be tuned on test.

## Required evaluations

Report aggregate and per-region:

* clean uncertainty/error Pearson and Spearman;
* uncertainty change versus measured degradation;
* RGB-only, thermal-only, and joint corruption response;
* target-region response and unrelated-region response;
* ranking accuracy and pair counts;
* calibration slope/intercept or monotonicity;
* uncertainty mean, standard deviation, and clamp-hit rate;
* subject-level correlations and bootstrap confidence intervals;
* repeated-corruption target sign agreement;
* runtime and memory.

Evaluate validation after every epoch. Select checkpoints using a pre-registered
multi-criterion rule, not ranking accuracy alone. Evaluate test only once after
selection.

## Individual-region analysis

For each of the four regions, report target distributions, positive and
negative degradation fractions, target reliability, modality-specific response,
and corruption response at held-out strengths. A global metric must not hide a
consistently failing region.

## Files to create later

The implementation phase should create new files only:

* `e3_e_build_corruption_features.py` — generate and validate paired clean /
  corrupted regional feature artifacts;
* `e3_e_objectives.py` — implement the bounded/probabilistic ranking and
  degradation terms without changing `e3_d_objectives.py`;
* `e3_e_train.py` — uncertainty-head-only controlled training;
* `e3_e_evaluate.py` — clean, corruption, per-region, and subject evaluation;
* `e3_d_objective_diagnostic/e3_e/` — target metadata, checkpoints, histories,
  and result tables.

No file above is created or executed by this Step 1 investigation.
