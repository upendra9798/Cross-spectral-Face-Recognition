# E3-E ranking-scale analysis

## Scope

This analysis is mathematical and uses the unchanged implementation in
`D:\RP\e3_d_objectives.py`. No objective, checkpoint, or model was modified.
No training was launched.

## Current loss

For a valid regional pair, let

* `s = sign(d_left - d_right)`;
* `Delta_u = u_left - u_right`;
* `m = 0` in the current E3-D implementation.

The current pairwise loss is:

```text
L_pair = softplus(m - s * Delta_u)
       = log(1 + exp(m - s * Delta_u))
```

The implementation averages all valid unordered region pairs and excludes
pairs whose absolute distance difference is at or below the fixed threshold.

## Scale properties

### Global offset invariance

For any constant `c`:

```text
(u_left + c) - (u_right + c) = u_left - u_right
```

Therefore the loss and all ranking gradients are unchanged by a global
uncertainty offset. The ranking objective cannot identify an absolute
uncertainty origin.

### Unbounded difference pressure

For a correctly ordered pair, `s * Delta_u > m`, and:

```text
dL/dDelta_u = -s * sigmoid(m - s * Delta_u)
```

The derivative remains negative for every finite correctly ordered
difference. The loss decreases continuously as the correctly signed
difference increases and approaches zero asymptotically. Thus, absent a
separate scale constraint, the objective has no finite optimum for the
uncertainty differences.

The current head clamps each modality output to `[-4, 4]`; the combined output
can therefore span at most `[-8, 8]`, and a pairwise combined difference can
reach at most 16. The clamp supplies an implementation bound, not a
statistically identified scale. It can also create boundary pressure when the
ranking loss keeps rewarding larger differences.

### Margin behavior

The current default margin is zero. Even with `m > 0`, the loss still rewards
larger correctly signed differences after the margin is exceeded; it only
shifts the point at which the loss becomes small. A margin does not identify
an absolute uncertainty scale or prevent output dispersion.

### Why standard deviation increased in E3-D4

The observed combined uncertainty standard deviation increased from
`0.088973` at epoch 1 to `0.544658` at epoch 10 while ranking accuracy rose
from `0.756190` to `0.773333`. This is consistent with the loss reducing
remaining ranking violations by expanding correctly signed pairwise
differences. It is not evidence of calibration. The global offset remains
free, and the absolute spread is selected by optimization dynamics,
initialization, head parameterization, and the output clamp rather than by a
calibration target.

## Bounded and probabilistic alternatives

### A. Bounded pairwise margin loss

One option is to define a bounded latent score:

```text
z = tanh(raw_score / temperature)
L = softplus(m - s * (z_left - z_right))
```

This limits pairwise differences and reduces runaway scale. It remains
offset-invariant in the latent pairwise sense and still does not calibrate an
absolute uncertainty level. A fixed temperature is required for comparable
gradients, and saturation can produce vanishing gradients.

### B. Probabilistic Bradley-Terry loss

Use the probability that the left region is less reliable:

```text
p(left worse) = sigmoid((u_left - u_right) / tau)
L_BT = -log sigmoid(s * (u_left - u_right) / tau)
```

This is the same logistic pairwise family as the current zero-margin
softplus loss, with an explicit temperature `tau`. A fixed temperature
controls gradient scale but does not solve global offset invariance or
absolute-scale identifiability. It still drives separable pairs toward
probability one.

### C. Calibrated probabilistic ranking

To obtain an identifiable scale, the pairwise term must be paired with an
absolute target or anchor, for example a train-only normalized degradation
target, a bounded calibrated distance target, or a zero-mean and variance
constraint. The anchor must be defined without validation or test data.
Without that additional term, changing from softplus to Bradley-Terry is not
enough.

## Conclusion

The existing ranking loss is useful for relative regional ordering, but it
cannot by itself produce calibrated absolute reliability. A future design
should use ranking only as an auxiliary ordinal term and add a measured,
bounded, train-only normalized target that identifies scale.
