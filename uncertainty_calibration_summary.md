# Uncertainty calibration investigation

- Full E3 history: `Experiment_3_Uncertainty_history.csv`
- Validation samples: 176
- Regions: left_eye, right_eye, nose, mouth
- Clean pooled uncertainty standard deviation: 0.096335
- Clean reliability standard deviation: 0.000095
- Conclusion: **C. Not yet learning meaningful regional reliability.**

## Clean distribution summary

```text
          rgb_log_uncertainty                               thermal_log_uncertainty                               reliability                              
                         mean       std       min       max                    mean       std       min       max        mean       std       min       max
region                                                                                                                                                     
left_eye            -0.528198  0.000513 -0.529429 -0.526309               -0.335328  0.000617 -0.336858 -0.333594    0.250001  0.000113  0.249669  0.250256
mouth               -0.527810  0.000953 -0.529874 -0.525277               -0.335585  0.000713 -0.337193 -0.333800    0.249968  0.000103  0.249602  0.250226
nose                -0.527979  0.000974 -0.529666 -0.525331               -0.335624  0.000759 -0.337239 -0.333738    0.250020  0.000071  0.249835  0.250188
right_eye           -0.528246  0.000568 -0.529727 -0.526547               -0.335317  0.000525 -0.336783 -0.334012    0.250010  0.000078  0.249753  0.250178
```

## Correlation summary

Pearson and Spearman correlations compare combined predicted uncertainty with observed regional RGB-thermal cosine distance.

```text
condition  strength    region  pearson_uncertainty_distance  spearman_uncertainty_distance  samples
    clean       0.0  left_eye                     -0.051182                      -0.133707      176
    clean       0.0 right_eye                      0.117718                      -0.016206      176
    clean       0.0      nose                      0.051507                      -0.048654      176
    clean       0.0     mouth                     -0.094899                      -0.188851      176
 left_eye       3.0  left_eye                     -0.028696                      -0.155947      176
 left_eye       3.0 right_eye                      0.144845                      -0.029943      176
 left_eye       3.0      nose                      0.068931                      -0.095060      176
 left_eye       3.0     mouth                     -0.034368                      -0.197166      176
 left_eye       7.0  left_eye                     -0.043296                      -0.167645      176
 left_eye       7.0 right_eye                      0.145456                      -0.028706      176
 left_eye       7.0      nose                      0.073575                      -0.094067      176
 left_eye       7.0     mouth                     -0.027297                      -0.189865      176
 left_eye      15.0  left_eye                     -0.045552                      -0.167091      176
 left_eye      15.0 right_eye                      0.145497                      -0.029733      176
 left_eye      15.0      nose                      0.074405                      -0.091880      176
 left_eye      15.0     mouth                     -0.026239                      -0.190007      176
right_eye       3.0  left_eye                     -0.009148                      -0.130387      176
right_eye       3.0 right_eye                      0.108364                      -0.042788      176
right_eye       3.0      nose                      0.085327                      -0.071314      176
right_eye       3.0     mouth                     -0.044756                      -0.157627      176
right_eye       7.0  left_eye                     -0.000344                      -0.129828      176
right_eye       7.0 right_eye                      0.098494                      -0.050863      176
right_eye       7.0      nose                      0.099488                      -0.073403      176
right_eye       7.0     mouth                     -0.032540                      -0.156321      176
right_eye      15.0  left_eye                      0.000678                      -0.127013      176
right_eye      15.0 right_eye                      0.097394                      -0.052435      176
right_eye      15.0      nose                      0.101581                      -0.071695      176
right_eye      15.0     mouth                     -0.030931                      -0.157240      176
     nose       3.0  left_eye                     -0.082549                      -0.178446      176
     nose       3.0 right_eye                     -0.005086                      -0.092333      176
     nose       3.0      nose                     -0.060308                      -0.184730      176
     nose       3.0     mouth                     -0.248542                      -0.337819      176
     nose       7.0  left_eye                     -0.091864                      -0.177970      176
     nose       7.0 right_eye                     -0.029937                      -0.121268      176
     nose       7.0      nose                     -0.096727                      -0.220362      176
     nose       7.0     mouth                     -0.258140                      -0.333956      176
     nose      15.0  left_eye                     -0.094195                      -0.176051      176
     nose      15.0 right_eye                     -0.034466                      -0.127116      176
     nose      15.0      nose                     -0.104500                      -0.230298      176
     nose      15.0     mouth                     -0.259508                      -0.336151      176
    mouth       3.0  left_eye                      0.163816                       0.068252      176
    mouth       3.0 right_eye                      0.084365                      -0.015598      176
    mouth       3.0      nose                     -0.143307                      -0.168321      176
    mouth       3.0     mouth                     -0.112749                      -0.163044      176
    mouth       7.0  left_eye                      0.162139                       0.091011      176
    mouth       7.0 right_eye                      0.071457                      -0.005087      176
    mouth       7.0      nose                     -0.150954                      -0.137310      176
    mouth       7.0     mouth                     -0.137962                      -0.166557      176
    mouth      15.0  left_eye                      0.161173                       0.095894      176
    mouth      15.0 right_eye                      0.069239                      -0.001653      176
    mouth      15.0      nose                     -0.152726                      -0.132523      176
    mouth      15.0     mouth                     -0.143174                      -0.166336      176
```

## Interpretation

The corruption rows use the same validation pairs as the clean rows. The current heteroscedastic objective rewards reducing log uncertainty until it is balanced by the regional distance term, but it has no calibration target. The observed near-zero uncertainty-distance correlations and near-uniform reliability therefore do not support reliability-weighted alignment yet. A corruption-supervised auxiliary objective would be a separate controlled experiment, not an implicit change to this phase.

## Gradient and objective diagnostics

Both uncertainty heads received non-zero gradients on a training batch after
loading the best checkpoint. Final-layer gradient norms were 1.1045 (RGB) and
0.7583 (thermal); output-bias norms were 0.9966 for both heads. The heads are
updating, but their outputs remain nearly region-invariant.

The validation uncertainty component decreased from `-0.2661` to `-0.8603`.
Clean regional matching distance had mean `0.001367` and standard deviation
`0.000475`, while reliability remained essentially uniform. Targeted blur
changed matching distance slightly but did not consistently increase predicted
uncertainty; targeted combined-uncertainty changes were negative for all tested
regions and strengths.