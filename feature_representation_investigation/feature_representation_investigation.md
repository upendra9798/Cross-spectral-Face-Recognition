# E3-C feature representation investigation

The 7x7 result uses the preserved best E3 checkpoint. The 14x14 and 28x28 probes use frozen EdgeFace backbone maps, a new 1x1 projection to 100 dimensions, and the unchanged E3 uncertainty matching objective. Their probe checkpoints are separate and do not overwrite existing artifacts.

```text
resolution  clean_distance_mean  clean_distance_std  rgb_uncertainty_mean  rgb_uncertainty_std  thermal_uncertainty_mean  thermal_uncertainty_std  reliability_std  clean_rgb_distance_pearson  clean_rgb_distance_spearman  clean_thermal_distance_pearson  clean_thermal_distance_spearman   seconds  seconds_per_batch  peak_gpu_memory_mb  trainable_probe_parameters
       7x7             0.001367            0.000475             -0.528059             0.000799                 -0.335463                 0.000673         0.000095                    0.092257                     0.036251                       -0.111709                        -0.117424 22.837353          22.837353                 NaN                      144442
     14x14             0.004364            0.004326             -0.151546             0.000816                 -0.196586                 0.002493         0.000604                   -0.162186                    -0.301826                        0.382803                        -0.086704 13.030207          13.030207                 NaN                       13030
     28x28             0.012318            0.005504             -0.339471             0.001248                  0.056856                 0.001735         0.000459                   -0.185078                    -0.196265                       -0.744388                        -0.718105  9.287242           9.287242                 NaN                        9830
```

Interpretation must be based on the per-resolution CSV files. This investigation does not implement reliability-weighted alignment.

## Clean-versus-corrupted analysis

The following values are mean changes from clean validation samples to the
three blur strengths averaged together, grouped by the region that was
corrupted. Positive uncertainty change would be the expected direction for a
useful corruption-sensitive score:

| Resolution | Left eye | Right eye | Nose | Mouth |
|---|---:|---:|---:|---:|
| 7x7 | -0.000088 | -0.000072 | -0.000064 | -0.000079 |
| 14x14 | -0.000176 | -0.000167 | -0.000252 | -0.000334 |
| 28x28 | +0.000581 | +0.000457 | +0.000539 | +0.000553 |

These changes are small and are not accompanied by consistent matching-distance
increases. The higher-resolution probes therefore show a measurable change in
raw uncertainty scale, but not demonstrated region-specific reliability.

## Conclusion

E3-C does not provide evidence that increasing the representation from 7x7 to
14x14 or 28x28 improves regional uncertainty discrimination. The 7x7 baseline
has reliability standard deviation `0.000095`; 14x14 and 28x28 increase this
only to `0.000604` and `0.000459`, respectively, while the correlations are
inconsistent in sign and magnitude. The strongest reported correlation is
28x28 thermal Pearson `-0.744`, which is inverse rather than the expected
positive relationship. The 28x28 blur response is positive but only about
`0.00046`-`0.00058` in combined log uncertainty, whereas the 7x7 and 14x14
responses move in the opposite direction.

This is a negative representation result, not evidence for Phase 4. The
current collapse remains primarily attributable to the uncertainty objective,
with feature resolution a secondary limitation. Clamp saturation was not
tested as a changed factor here and remains excluded based on the prior E3-B
audit. All 14x14 and 28x28 probe checkpoints are separate artifacts; the
original E3 checkpoint and training artifacts were preserved.