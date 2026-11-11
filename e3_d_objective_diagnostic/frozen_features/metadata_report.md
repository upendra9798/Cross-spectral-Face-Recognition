# E3-D0 frozen regional feature extraction

- Checkpoint: `D:\RP\Experiment_3_Uncertainty_best.pt`
- Checkpoint SHA-256: `821c312dc332ecbfed64272b50e867bf72dff2bc341518df284226228c86fa03`
- Device: `cpu`
- Seed: `42`
- Batch size: `64`
- Model parameters modified: no
- Checkpoint modified: no
- Feature shape per sample: `[4, 100]`
- Distance shape per sample: `[4]`
- Feature normalization: L2 normalization over the feature dimension.
- Distance: `1 - cosine_similarity(rgb, thermal)` per region.
- Invalid samples: preserved and flagged in `valid_regions`; no invalid regions were silently dropped.

## Split extraction

| Split | Samples | Subjects | Invalid regions | File | Size (bytes) | Seconds |
|---|---:|---:|---:|---|---:|---:|
| train | 1772 | 100 | 0 | train_features.pt | 6647057 | 12.758 |
| val | 176 | 10 | 0 | val_features.pt | 662459 | 1.291 |
| test | 565 | 32 | 0 | test_features.pt | 2121158 | 4.868 |

## Validation

All artifacts were reloaded successfully. Shapes, finite values, saved distances, identifier counts, landmark shapes, region-validity shapes, subject-disjoint splits, pair-disjoint splits, checkpoint bytes, and model parameters were verified.