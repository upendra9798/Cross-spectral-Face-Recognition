# E3-G leakage audit

## Controls

| Control | Status |
|---|---|
| Original E3 checkpoint preserved | verified by SHA-256 before/after extraction |
| E3-D0 regional artifacts preserved | read-only inputs |
| Backbone and Adapter training | not performed |
| Confidence network training | not performed |
| Original regional extractor changed | no |
| Subject-disjoint source splits | verified by E3-D0 metadata |
| Pair construction within split | enforced |
| Genuine labels | same subject ID only |
| Impostor labels | different subject IDs only |
| Impostor sampling | deterministic, seed 42 |
| Calibration fitting | train subjects only |
| Validation use | method/threshold decisions only |
| Test use | final evaluation only |
| Test threshold tuning | not performed |

## Dependence caveat

One pair per source row is reproducible, but pairs are not independent:
multiple rows share subjects, and RGB or thermal images can recur across
comparisons in larger protocols. Aggregate confidence intervals must therefore
resample subjects, not individual pair rows. The current baseline CSVs report
point statistics and explicitly do not claim independent-observation
confidence intervals.

## Identity and acquisition shortcuts

The calibration features may encode modality mismatch, subject acquisition
conditions, or region-specific artifacts. Agreement is not independent
correctness evidence. A future fitted confidence model must compare distance
only, distance plus norms, agreement, consistency, and combined features under
subject-disjoint validation.

## Reproducibility

`e3_g_reproducibility.json` records the seed, checkpoint hash, feature root,
pair construction rule, and no-training status. The audit uses
`shuffle=False`, CPU inference, and the existing split ordering.
