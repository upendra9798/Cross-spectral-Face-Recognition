# E3-F recommended research direction

## Recommendation

Proceed with **region-wise verification confidence calibration with
selective-risk evaluation**, using cross-modal agreement and optional
transformation-consistency features as inputs rather than treating either as
ground-truth reliability.

This is the most defensible next direction because:

1. It identifies absolute confidence scale from recognition outcomes.
2. It is compatible with the frozen EdgeFace-XS, Transformer Adapter, and
   four-region features.
3. It directly tests whether a regional score predicts verification error.
4. It supports operational risk-coverage analysis.
5. It does not require fabricated degradation or uncertainty labels.
6. It exposes the jointly-wrong-agreement failure mode explicitly.

## What this does and does not claim

The output is calibrated **pairwise regional match confidence**. It is not
automatically an intrinsic RGB quality score, thermal quality score, or
epistemic uncertainty estimate. Independent-modality quality should be
reported separately, if investigated, using transformation-consistency
diagnostics and held-out recognition outcomes.

## Proposed staged design

### Stage 1: non-parametric baseline

Evaluate whether `d_rgb_thermal(region)` alone predicts genuine/impostor
verification outcomes. Report per-region ROC/PR, Brier score, ECE, calibration
plots, and risk-coverage.

### Stage 2: small calibrated confidence head

Fit a small region-aware calibration model on frozen features using:

* cross-modal regional distance;
* RGB and thermal feature norms;
* optional cross-modal agreement features;
* optional transformation-consistency features;
* region identifier only if it improves subject-disjoint calibration.

Do not unfreeze the backbone or Adapter initially.

### Stage 3: selective regional decision

Measure whether rejecting low-confidence regions reduces verification error at
fixed coverage. Compare:

* global score only;
* equal regional weighting;
* calibrated regional confidence;
* oracle region selection as an upper bound.

## Why not the prior degradation target?

E3-E4 found 60.6% average mixed-sign samples across stochastic draws, only
41.7% positive target-region subject-bootstrap intervals, and stronger local
responses in only 33.3% of tested rows. Changing loss weights or training
longer would not solve target validity.

## Research contribution potential

The defensible contribution is not “uncertainty heads improve recognition”
without evidence. A stronger contribution would be a rigorous study of
region-wise calibrated verification confidence under RGB-thermal mismatch,
including failure analysis of agreement, consistency, and selective rejection.
