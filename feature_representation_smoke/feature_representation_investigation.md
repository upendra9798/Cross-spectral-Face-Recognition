# E3-C feature representation investigation

The 7x7 result uses the preserved best E3 checkpoint. The 14x14 and 28x28 probes use frozen EdgeFace backbone maps, a new 1x1 projection to 100 dimensions, and the unchanged E3 uncertainty matching objective. Their probe checkpoints are separate and do not overwrite existing artifacts.

```text
resolution  clean_distance_mean  clean_distance_std  rgb_uncertainty_mean  rgb_uncertainty_std  thermal_uncertainty_mean  thermal_uncertainty_std  reliability_std  clean_rgb_distance_pearson  clean_rgb_distance_spearman  clean_thermal_distance_pearson  clean_thermal_distance_spearman   seconds  seconds_per_batch  peak_gpu_memory_mb  trainable_probe_parameters
       7x7             0.001367            0.000475             -0.528059             0.000799                 -0.335463                 0.000673         0.000095                    0.092257                     0.036251                       -0.111709                        -0.117424 17.336541          17.336541                 NaN                      144442
     14x14             0.004699            0.004830             -0.018405             0.001386                 -0.049504                 0.003565         0.000945                    0.165321                    -0.008376                        0.641920                         0.534419 11.164958          11.164958                 NaN                       13030
     28x28             0.012010            0.004709              0.169896             0.000805                  0.125653                 0.001484         0.000324                   -0.093867                    -0.107251                        0.098414                         0.069027  9.171550           9.171550                 NaN                        9830
```

Interpretation must be based on the per-resolution CSV files. This investigation does not implement reliability-weighted alignment.