# E3-E4 recommendation

**Classification: Not sufficiently supported.**

The stochastic target construction is reproducible for fixed seeds, but the measured target is not sufficiently repeatable or region-specific for uncertainty-head training with this operator.

Evidence against immediate training:
1. Most samples have mixed degradation signs across the five draws.
2. Subject-level confidence intervals are positive for fewer than half of target rows.
3. Local responses exceed unrelated responses in only a minority of tested rows.
4. The current Gaussian-noise proxy is not established as representative of sensor degradation.

If a future operator is evaluated, it must satisfy the following before training:
1. Preserve signed degradation and use robust target regression.
2. Normalize targets from training subjects only.
3. Require target-region versus unrelated-region specificity.
4. Keep ranking as an auxiliary bounded/temperature-controlled term.
5. Treat RGB-only, thermal-only, and both-modality targets separately.
6. Establish stronger repeatability with a scientifically justified operator.

No uncertainty training or objective modification was performed.
