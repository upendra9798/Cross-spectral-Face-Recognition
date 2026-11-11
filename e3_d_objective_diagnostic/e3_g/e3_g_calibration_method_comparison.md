# E3-G calibration method comparison

## Definitions

For a pair, `correct = 1` means genuine same-subject RGB-thermal verification
and `correct = 0` means an impostor pair. Confidence is a probability
`p(correct=1)`. It is pairwise verification confidence, not an intrinsic
uncertainty label for either modality.

The audit compares each global or regional similarity score with:

1. **Raw standardized sigmoid**: a descriptive non-fitted baseline. The
   training mean and standard deviation standardize the score; it is not
   claimed to be calibrated.
2. **Logistic calibration**: fits `p = sigmoid(a*s+b)` on training subjects
   only by deterministic gradient descent.
3. **Temperature scaling**: fits `p = sigmoid(s/T)` on training subjects,
   retaining no intercept. It is appropriate only when the input score is
   already a meaningful logit-like quantity; cosine similarity is not
   automatically a logit, so this is a diagnostic comparison.

Validation subjects are used for selecting the calibration family and any
operating threshold. Test results are final and are not used in fitting or
tuning.

## Metrics

* Brier score: mean squared probability error.
* Negative log-likelihood: binary log loss.
* Expected Calibration Error: confidence/empirical accuracy gap over ten
  predeclared bins.
* ROC-AUC: ranking discrimination, independent of calibration.
* TAR@FAR 0.01 and 0.10: genuine acceptance at an impostor false-accept
  operating point.

Calibration metrics are reported in `e3_g_calibration_statistics.csv`, while
score distribution and discrimination statistics are in
`e3_g_regional_score_statistics.csv`.

## Interpretation guardrails

Lower ECE or Brier score alone does not establish a useful regional
reliability estimate. Results must also preserve verification discrimination,
generalize to held-out subjects, and improve selective risk at a declared
coverage. Regional scores must be reported separately from the global score.
