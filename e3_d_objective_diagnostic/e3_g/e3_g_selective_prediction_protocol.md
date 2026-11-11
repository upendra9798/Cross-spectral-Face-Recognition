# E3-G selective prediction protocol

## Decision quantities

For a verification pair, correctness is the binary genuine/impostor label.
Confidence is the calibrated probability of a genuine match. A retained pair
is one whose confidence is at least the threshold induced by a requested
coverage. Selective risk is the impostor fraction among retained pairs:

```text
coverage(t) = retained_pairs(t) / all_pairs
risk(t) = mean(1 - correctness | retained_pairs(t))
```

The confidence ranking is generated without using test labels. The audit
reports the full risk-coverage curve and its trapezoidal area. Lower risk at
the same coverage is better; a trivial rule that retains almost nothing is
not sufficient evidence.

## Global versus regional rejection

The baseline compares global similarity confidence with each of the four
regional similarity confidences. A future region-combination experiment must
pre-register how regional evidence is combined. The initial safe options are:

* reject a pair if any required region is below its validation-selected
  threshold;
* retain a pair if the mean calibrated regional confidence exceeds its
  threshold;
* report each region independently without silently imputing missing evidence.

No missing regional evidence occurs in the E3-D0 artifacts, but invalid-region
flags must be propagated if later artifacts contain them. Missing evidence is
not evidence of high confidence.

## Comparison requirement

Global and regional rejection must use the same pair lists, labels, subject
split, and coverage grid. Improvement is claimed only when measured against
the same test pairs and a predeclared operating coverage. Subject-level
bootstrap intervals should be added before making a final research claim.
