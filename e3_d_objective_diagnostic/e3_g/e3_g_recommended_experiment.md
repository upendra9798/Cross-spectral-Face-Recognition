# E3-G recommended experiment

## Recommendation

Approve a later, small statistical confidence-model experiment only after
reviewing the generated baseline CSVs. Do not yet train a confidence network.
The first implementation should fit a region-wise logistic calibration model
on frozen pair-level scores, with validation-selected calibration and
selective-risk thresholds.

## Required sequence

1. Reproduce the E3-G pair CSVs and verify subject/pair checks.
2. Fit distance-only or similarity-only calibration on training subjects.
3. Select calibration family and operating coverage on validation subjects.
4. Evaluate once on test subjects.
5. Add feature-consistency inputs only as a pre-registered ablation.
6. Compare global, region-wise, and combined regional rejection on identical
   pairs.

## Advancement criteria

Advance only if, on held-out subjects, a region-wise score:

* improves or preserves ROC-AUC and TAR at declared FAR values;
* improves Brier score, NLL, and ECE against the uncalibrated baseline;
* reduces selective risk at a predeclared coverage;
* does not fail materially in one facial region;
* remains useful when cross-modal agreement is ablated;
* has subject-level uncertainty intervals.

These are decision criteria, not expected results.

## Current scope boundary

This phase provides protocol, baseline scores, calibration utilities, and
leakage controls. It does not establish that regional confidence improves
recognition, and it does not authorize modifying the Transformer Adapter,
regional extractor, checkpoints, or alignment objective.
