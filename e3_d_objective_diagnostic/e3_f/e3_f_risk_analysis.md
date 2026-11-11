# E3-F risk analysis

## Cross-cutting risks

### Reliability versus agreement

Two modalities can agree because both encode the same wrong identity or the
same preprocessing artifact. Cross-modal agreement is therefore a matching
signal, not independent evidence of correctness.

### Reliability versus uncertainty

An uncertainty value is useful only if it predicts an outcome or a
distributional error. Dispersion, variance, or standard deviation without
calibration is not reliability.

### Reliability versus consistency

Transformation consistency measures local representation stability. It does
not test whether the stable representation is identity-correct.

### Subject leakage

Regional confidence models can memorize subject or acquisition signatures.
All pair construction, normalization, calibration, and checkpoint selection
must preserve subject-disjoint splits.

### Region imbalance

The four regions have different distance distributions and different
cross-region coupling. A global confidence metric can hide failure in the
mouth or nose region. All results must be reported per region.

### Pair construction

Verification confidence requires genuine and impostor comparisons. Positive
and negative pair generation must be explicit, balanced or reweighted, and
disjoint by subject between fitting and calibration.

### Calibration leakage

Temperature or isotonic calibration fitted on validation outcomes cannot be
used to claim test-time generalization if the same validation set selected
the model or transformation choices. The protocol must pre-register the
calibration role of each split.

## Candidate-specific risks

| Candidate | Main shortcut | Failure mode | Required control |
|---|---|---|---|
| Feature consistency | learn the transformation family | stable but wrong features | calibrate against identity outcomes |
| Cross-modal agreement | exploit modality alignment | jointly incorrect agreement | genuine/impostor calibration and counterexamples |
| Verification confidence | memorize subject/acquisition | inflated validation confidence | subject-disjoint episodic construction |
| Probabilistic embeddings | variance collapse or inflation | unidentified scale | likelihood checks, variance floors, calibration |
| Selective prediction | reject too many samples | trivial high-precision/low-coverage rule | risk-coverage and minimum-coverage constraints |

## E3-specific risks

E3-D4 showed that ranking can improve regional ordering while clean
uncertainty/error correlation remains near zero. E3-E3 and E3-E4 showed
signed degradation, modality asymmetry, cross-region responses, and inadequate
stochastic target repeatability. These results argue against rescuing the
existing uncertainty head with a different loss weight or longer training.

The main unresolved question is whether the SF-TL54 labels and pair structure
provide enough genuine/impostor regional comparison outcomes to calibrate a
useful confidence score without leakage. That question should be answered
before implementing a new uncertainty objective.
