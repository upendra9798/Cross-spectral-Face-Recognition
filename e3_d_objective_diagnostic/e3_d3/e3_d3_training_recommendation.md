# E3-D3 training recommendation

No longer training was launched.

## Evidence

The ranking-only p50 checkpoint is the strongest short-run discriminator:
it has higher ranking accuracy and a larger positive corruption response than
the combined p50 checkpoint. However, its global clean correlations remain
near zero, so it has not demonstrated calibration.

The combined p50 normalized component has a gradient cosine comparison against
the ranking component in `e3_d3_gradient_comparison.csv`. These values should
be interpreted as directional diagnostics, not proof of suppression.

## Proposed next experiment

- Train only the uncertainty heads for 10 epochs initially.
- Keep the p50 threshold `0.000161975622` and ranking weight unchanged.
- Keep Adam, learning rate `0.001`, seed `42`, batch size `64`, and shuffle disabled.
- Evaluate after every epoch.
- Select checkpoints by validation combined Spearman correlation, subject to:
  - positive Pearson and Spearman,
  - ranking accuracy above the three-epoch p50 reference `0.756190`,
  - positive uncertainty change for the directly corrupted region at held-out
    blur strengths,
  - no increase in clamp-hit rate,
  - no collapse of reliability variation.
- Use patience 3 early stopping on validation Spearman.
- Do not use test results for selection.

The current three-epoch ranking p50 checkpoint remains the reference. This
recommendation is provisional and does not authorize the longer run or Phase 4.

## D3 evidence summary

On the identical initial heads and a 256-sample frozen training batch:

- Normalized fixed-scale parameter-gradient norm: `2.469675`
- Ranking p50 parameter-gradient norm: `0.006361`
- Combined p50 parameter-gradient norm: `2.469663`
- Output-gradient cosine(normalized, ranking): `0.189688`
- Output-gradient cosine(normalized, combined): `0.839290`
- Output-gradient cosine(ranking, combined): `0.693016`

The combined gradient magnitude is therefore dominated by the normalized term
under the current unchanged ranking weight. This is consistent with ranking
behavior being diluted, but does not by itself prove that changing the weight
would improve calibration.

For the ranking p50 checkpoint, mean per-region validation/test ranking accuracy
was `0.755889` / `0.758671`; for combined p50 it was `0.702044` / `0.719481`.
Mean per-region combined Pearson correlation was `-0.000153` / `-0.018260` for
ranking p50 and `-0.068087` / `-0.165697` for combined p50. These results show
ordering generalization without corresponding clean error calibration.

The combined p50 nose response is negative for every tested blur strength
(`sigma=3,7,15`) even though the nose matching distance increases. Ranking p50
has positive combined uncertainty changes for all four directly corrupted
regions and all three blur strengths, but this corruption response still does
not establish clean uncertainty calibration.
