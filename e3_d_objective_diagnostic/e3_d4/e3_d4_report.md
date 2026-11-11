# E3-D4 longer ranking-only uncertainty experiment

- Reference checkpoint: `D:\RP\e3_d_objective_diagnostic\e3_d2\e3_d2_checkpoints\ranking_p50_heads.pt`
- Reference checkpoint file SHA-256: `8ccf15fd340fc96e2b5b61cc0607b5e9213c9383bc6fc53f7a54e2e70c5519b0`
- Initial uncertainty-head state hash: `16f1b79a3734440b211dd07f99625cd2a4b64b527aa9a531d0d8ad79a13b398a`
- Final uncertainty-head state hash: `0648da9ecf2fbe82df3778a849bdd15bd697056de20475ac4b1ae6d1f67a124f`
- Threshold: `0.000161975622`
- Objective: pairwise ranking only
- Training: 10 epochs maximum, Adam, learning rate 0.001, batch size 64, seed 42, shuffle disabled
- Test data was not used for training or checkpoint selection.

## Selection result

Eligible epochs: `[]`
Selected epoch: `None`

Eligibility required positive global Pearson and Spearman, ranking accuracy at least 0.756190, positive targeted corruption response for all regions, no negative per-region Pearson values, and zero clamp hits.

If no epoch is selected, the experiment did not produce a successful checkpoint under the pre-registered criteria.

Per-epoch, regional, subject and corruption metrics are saved in the requested CSV files. Subject bootstrap summaries contain 95% confidence intervals over subject-level correlations.