# E3-F candidate formulations

## Notation

For sample `i` and region `r`:

* `z_rgb(i,r)`, `z_th(i,r)` are regional embeddings;
* `d(i,r)` is the cross-modal regional distance;
* `y(i,j)` is the verification label for a paired comparison;
* `q(i,r)` is a regional quality/reliability variable;
* `c(i,j,r)` is calibrated regional match confidence.

Reliability is a latent property of expected future recognition performance.
Uncertainty is a predictive distribution or dispersion quantity. Feature
consistency measures sensitivity to a transformation. Matching confidence is
the probability that a particular comparison is correct. They are related but
not interchangeable.

## A. Feature-consistency reliability

For controlled transformations `T_k`:

```text
v(i,r) = mean_k || z(i,r) - z(T_k(i),r) ||^2
q(i,r) = g(v(i,r), feature_norm(i,r), region_validity(i,r))
```

Inputs: one modality image, landmarks, and a pre-registered transformation
family. Output: modality-specific transformation sensitivity.

Supervision: none or self-supervised consistency. Ground-truth uncertainty
labels are not required. Inference can be independent per modality.

Limitations: a representation can be invariant to an identity-destroying
transformation, or sensitive to a harmless one. Consistency is not correctness.
The transformation family can become a shortcut. Calibration must use held-out
identity outcomes, not consistency loss alone.

Compatibility: high with frozen regional features. Cost: one or more extra
forward passes per image. Recommended use: auxiliary quality feature and
diagnostic, not the sole reliability target.

## B. Cross-modal agreement

```text
a(i,r) = -d(z_rgb(i,r), z_th(i,r))
q(i,r) = g(a(i,r), ||z_rgb||, ||z_th||, region metadata)
```

Inputs: paired RGB-thermal regional embeddings. Output: pair-specific
agreement or confidence.

Supervision: verification labels are required for calibration; no explicit
uncertainty labels are required. Inference requires paired modalities and
cannot support independent RGB-only or thermal-only reliability without a
separate modality head.

Limitations: jointly incorrect representations can agree. Agreement also
mixes identity similarity, modality compatibility, and sample quality. The
E3 clean-distance correlation results show that agreement-derived distance is
not automatically a calibrated regional reliability target.

Compatibility: very high for paired inference. Cost: negligible after feature
extraction. Recommended use: an input to calibrated verification confidence,
not a standalone uncertainty estimate.

## C. Indirectly supervised verification confidence

Construct subject-disjoint regional verification episodes with genuine and
impostor labels:

```text
p_correct(i,j,r) = sigmoid(h_theta(
    d(i,j,r), a(i,j,r), q_rgb(i,r), q_th(j,r), region_id
))
L = binary_cross_entropy(p_correct, y)
```

The model predicts probability of a correct regional match, not an abstract
uncertainty. A calibration map can be fit after training using validation
subjects only, for example temperature scaling or isotonic calibration.

Inputs: paired regional features and comparison labels. Ground-truth
uncertainty labels are not required; recognition outcomes provide indirect
supervision. Inference is pair-specific and requires both modalities.

Advantages: absolute scale is identified by the binary outcome, evaluation
can use Brier score, expected calibration error, risk-coverage, AUROC, and
selective verification risk, and region-specific performance can be measured.

Risks: pair construction leakage, subject memorization, class imbalance, and
confidence learning shortcuts based on region identity or modality artifacts.
All episode construction and calibration must preserve subject disjointness.

Compatibility: high. Cost: low to medium for a small calibration head; no
backbone changes are required.

## D. Distributional/probabilistic regional embeddings

Predict a regional Gaussian:

```text
z_r ~ N(mu_r, diag(sigma_r^2))
```

Use a probabilistic matching score such as expected distance or a
mutual-likelihood score, with recognition supervision identifying the
variance. This follows the probabilistic-face-embedding direction.

Inputs: regional features from each modality. Ground-truth uncertainty labels
are not required if the likelihood/verification objective is well specified.
Inference can be single-modality for embedding uncertainty, but cross-modal
matching still needs both distributions.

Risks: variance can become numerically or statistically miscalibrated; the
objective must identify the variance; fixed frozen features may not contain
enough information for a useful variance head. Calibration under spectral
shift remains an empirical question.

Compatibility: medium to high, but it is a larger objective redesign than
post-hoc confidence calibration. Cost: medium.

## E. Selective regional prediction

Use a score `s(i,j,r)` to decide whether a regional comparison is accepted:

```text
coverage(t) = P(s >= t)
risk(t) = P(verification_error | s >= t)
```

The desired property is controlled risk at useful coverage, not merely high
uncertainty variance. A regional confidence score can be learned from
verification outcomes and calibrated on held-out subjects.

Inputs: pairwise features/scores and verification labels. Requires paired
inference for cross-modal decisions. It can support an operational reject
rule and directly tests whether reliability is useful.

Compatibility: high as an evaluation and decision layer. Cost: low.
Limitation: it does not by itself generate a new intrinsic modality-quality
estimate.
