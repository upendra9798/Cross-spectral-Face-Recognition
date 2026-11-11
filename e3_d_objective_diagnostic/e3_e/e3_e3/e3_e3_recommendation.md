# E3-E3 recommendation

## Classification

**Supported only with restrictions**

## Evidence

- 100.0% of target condition/region combinations had at least half of subjects with positive mean degradation.
- 94.4% had both subject-level bootstrap confidence intervals above zero and at least half of subjects positive.
- 94.4% retained bootstrap evidence after Benjamini-Hochberg correction across the full test family.
- 41.7% of target-region responses were monotonic non-decreasing across the tested blur strengths (5/12 region-by-mode target trends).
- RGB-only target responses were monotonic for 4/4 regions; thermal-only responses were monotonic for 0/4 regions; both-modality responses were monotonic for 1/4 regions.
- Negative and near-zero responses are present and are retained in the comprehensive statistics.
- Local target responses exceeded unrelated-region averages, but unrelated-region responses remained non-zero, so cross-region coupling must be modeled and monitored.
- Deterministic blur does not establish stochastic repeatability.

## Restrictions before training

- Generate multiple explicitly seeded stochastic corruption draws.
- Estimate target normalization constants from training data only.
- Use a robust loss and preserve signed degradation.
- Require target-region versus unrelated-region specificity checks.
- Report subject-level confidence intervals and FDR-adjusted tests.
- Retain bounded or temperature-controlled ranking as an auxiliary objective.

No uncertainty training or objective implementation was performed in E3-E3.
