# E3 uncertainty objective audit

The E3 objective is:

`L(u, d) = exp(-u) * d + u`

where `u` is combined log uncertainty and `d` is regional cosine distance.
Its exact derivative is:

`dL/du = 1 - exp(-u) * d`

Setting the derivative to zero gives the unconstrained optimum:

`u* = log(d)`

With the implementation bounds `[-4, 4]`, the optimum is
`clamp(log(d), -4, 4)`.

## Observed validation regime

- Mean clean regional distance: `0.001366639`
- Mean trained combined log uncertainty: `-0.863522`
- Unconstrained optimum at mean distance: `-6.595401`
- Bounded optimum at mean distance: `-4.000000`
- Derivative at the trained mean: `0.996759`

For the observed distance scale, `log(d)` is below the lower clamp. The
positive derivative at the trained mean therefore pushes optimization toward
the lower bound for every region. This objective does not distinguish regions
unless their distances differ by enough to produce materially different
`log(d)` values above the clamp. It explains the downward uncertainty trend
and does not by itself establish calibrated reliability.

The numerical grid is saved in `uncertainty_objective_audit.csv`. No model or checkpoint
was modified.
