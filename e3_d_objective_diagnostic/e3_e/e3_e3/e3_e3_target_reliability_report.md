# E3-E3 degradation target validity and robustness audit

## Scope

- Audited all 36 existing E3-E2 deterministic Gaussian-blur artifacts.
- Split: training only; 1,772 samples and 100 subjects.
- No new corruption draws were generated.
- No training, objective implementation, checkpoint modification, or Phase 4 work was performed.

## Statistical method

- Subject-level bootstrap samples: 4000.
- Bootstrap resampling unit: subject, not individual image.
- Two-sided bootstrap p-values were adjusted with Benjamini-Hochberg FDR across all 144 mode/condition/observed-region tests.
- Near-zero threshold: absolute delta_d < 1e-05.
- Effect size is mean delta_d divided by the clean-distance standard deviation for the observed region.

## Main measured results

- Target conditions with at least 50% positive subject means: 1.000.
- Target conditions with bootstrap CI entirely above zero and at least 50% positive subjects: 0.944.
- Target conditions surviving BH-FDR q < 0.05 with CI above zero: 0.944.
- Target-region strength trends monotonic non-decreasing across sigma 3, 7, 15: 0.417.

These summaries do not treat positive mean degradation alone as proof of a reliable target.
The full condition-level, subject-level, response-matrix, trend, and modality results are in the CSV files.

## Modality and strength findings

- RGB-only target-region responses were monotonic non-decreasing for 4/4 regions.
- Thermal-only target-region responses were monotonic non-decreasing for 0/4 regions; all four decreased over the tested strengths.
- Both-modality target-region responses were monotonic non-decreasing for 1/4 regions.
- The target-region bootstrap/FDR criterion was met by 34/36 target condition/region rows.
- Local target responses exceeded the mean unrelated-region response for each mode and tested strength, but unrelated-region responses remained non-zero.
- Therefore the target is modality-dependent and partially localized, not a direct scalar function of blur strength.

## Region specificity

The response matrix reports every corrupted-region/observed-region combination.
Cross-region changes are retained because regional feature extraction and the adapter can propagate effects across regions.
Localized supervision should therefore be evaluated using target-versus-unrelated response contrasts, not target positivity alone.

## Repeatability limitation

The source Gaussian blur is deterministic for fixed image, landmark, region, and sigma.
These artifacts do not establish stochastic repeatability, draw-to-draw variance, or target sign agreement under random corruption.
A future audit should generate multiple explicitly seeded stochastic draws per sample and condition.

## Classification

Degradation supervision is classified as: **Supported only with restrictions**.

Restrictions:
1. Use robust regression rather than treating delta_d as noise-free.
2. Preserve negative targets; do not clip or rectify them.
3. Normalize targets using training data only.
4. Require subject-level bootstrap and FDR-controlled evidence.
5. Monitor target-versus-unrelated region contrasts.
6. Establish stochastic repeatability before using degradation as the primary training target.
7. Keep bounded/temperature-controlled ranking as an auxiliary term rather than the absolute calibration mechanism.

## Output files

- `e3_e3_comprehensive_statistics.csv`
- `e3_e3_subject_bootstrap.csv`
- `e3_e3_region_response_matrix.csv`
- `e3_e3_strength_trend.csv`
- `e3_e3_modality_comparison.csv`
- `e3_e3_target_reliability_report.md`
- `e3_e3_recommendation.md`
