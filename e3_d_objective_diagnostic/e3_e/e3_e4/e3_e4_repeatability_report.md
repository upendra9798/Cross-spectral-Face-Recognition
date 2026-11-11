# E3-E4 stochastic corruption repeatability audit

- Split: training only (1,772 samples, 100 subjects).
- Conditions: RGB-only, thermal-only, both-modality; four regions; sigma 3, 7, 15.
- Independent draws per condition: 5.
- Seeds: 101, 202, 303, 404, 505.

## Operator

Localized additive Gaussian sensor/read noise was applied only inside the landmark-derived region box. Sigma values are interpreted as 8-bit intensity standard deviations and converted to normalized-image standard deviations by sigma / 127.5. This keeps the physical intensity scale comparable while preserving the observed RGB/thermal distribution differences. Values were clamped only to the existing normalized image range [-1, 1].

## Results

- Target-region conditions with at least 80% positive per-sample draws: 0.472.
- Target-region conditions with non-decreasing mean response across sigma levels: 0.500.
- Across all condition/observed-region rows, mean per-sample sign fractions were 22.3% consistently positive, 17.1% consistently negative, and 60.6% mixed.
- Only 41.7% of target-region rows had subject-bootstrap confidence intervals above zero.
- Local mean degradation exceeded the unrelated-region mean in only 33.3% of mode/region/strength rows.
- Negative, zero, and positive per-sample degradation values were preserved.
- Subject bootstrap summaries and FDR-adjusted sign tests are saved in the CSV outputs.

## Repeatability interpretation

Five draws provide an initial repeatability audit but do not establish population-level repeatability by themselves. The saved draw-level standard deviations and subject-level summaries must be used with confidence intervals.
The operator is a controlled sensor-noise proxy, not a claim that Gaussian noise represents all real RGB or thermal sensor degradation.

## Integrity

- Same seed uses a deterministic torch generator.
- Different seeds generate independent random noise tensors.
- Clean features, pairing, subjects, landmarks, and geometry are preserved.
- Model parameters were unchanged.
- The original checkpoint and prior E3-D0/E3-E2 artifacts were not modified.

## Files

- e3_e4_corruption_metadata.csv
- e3_e4_draw_statistics.csv
- e3_e4_subject_repeatability.csv
- e3_e4_sign_agreement.csv
- e3_e4_region_specificity.csv
- e3_e4_modality_comparison.csv
- e3_e4_strength_trend.csv
- e3_e4_operator_metadata.txt
- e3_e4_repeatability_report.md
- e3_e4_recommendation.md
