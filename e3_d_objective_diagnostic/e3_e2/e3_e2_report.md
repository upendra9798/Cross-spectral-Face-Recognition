# E3-E2 clean and corrupted regional feature construction

- Checkpoint: `D:\RP\Experiment_3_Uncertainty_best.pt`
- Checkpoint SHA-256: `821c312dc332ecbfed64272b50e867bf72dff2bc341518df284226228c86fa03`
- Split: `train` (1772 samples)
- Corruption modes: RGB-only, thermal-only, and both-modality
- Regions: left eye, right eye, nose, mouth
- Blur strengths: sigma 3, 7, 15
- Features: clean and corrupted regional tensors with shape `[N, 4, 100]`
- Negative degradation values were preserved without clipping or rectification.

## Target definitions

- `d_clean = 1 - cosine(rgb_clean, thermal_clean)`
- `d_corrupted = 1 - cosine(rgb_corrupted, thermal_corrupted)`
- `delta_d = d_corrupted - d_clean`
- RGB change: `1 - cosine(rgb_clean, rgb_corrupted)`
- Thermal change: `1 - cosine(thermal_clean, thermal_corrupted)`

## Repeatability

The implemented Gaussian blur is deterministic for fixed images, landmarks, region, and sigma. Therefore repeated identical draws would not measure stochastic corruption variability and were not fabricated. Future stochastic corruptions require repeated seeded draws and sign-agreement analysis.

## Validation

- All condition artifacts were reloaded and validated.
- Feature, distance, degradation, landmark, pair-key, and subject-ID shapes passed.
- Recomputed clean and corrupted distances matched saved values.
- Recomputed degradation matched `d_corrupted - d_clean`.
- All saved values were finite.
- Model parameters were unchanged.
- The original checkpoint was unchanged.
- Existing E3-D0 frozen artifacts were not overwritten.

## Files

- `clean_features.pt`
- `36` corruption condition artifacts
- `e3_e2_target_summary.csv`
- `e3_e2_subject_target_summary.csv`
- `e3_e2_metadata.json`
- `e3_e2_report.md`
