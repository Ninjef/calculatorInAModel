# Rank-normalized expected answer loss rescues result-space expected-cost discovery.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-rank-normalized-expected-loss-gate.md

Summary:

- Added `expected_answer_loss_cost_normalization=rank`, which replaces each prompt's forced-result NLLs with within-prompt ranks before the exact policy expectation. The full-grid Stage 0 diagnostic did not clear the gate: exact-vs-boundary result-proj cosine was only `0.049551` and upstream cosine was `0.002584`, weaker than an earlier contrastive-margin decoder sign flip that still failed Stage 1. Sampled PG remained aligned with the rank objective (`0.723444`/`0.719965`) but only barely aligned with the boundary target (`0.027483`/`0.016271`).

Questions:

- What did we learn about Rank-normalized expected answer loss rescues result-space expected-cost discovery?
- Has Rank-normalized expected answer loss rescues result-space expected-cost discovery been tested?
- Should we repeat Rank-normalized expected answer loss rescues result-space expected-cost discovery?
- What is the status of Rank-normalized expected answer loss rescues result-space expected-cost discovery?
- Why did Rank-normalized expected answer loss rescues result-space expected-cost discovery fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-rank-normalized-expected-loss-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run rank-normalized expected answer-loss Stage 1, or rank/scale transforms of the same full-enum expected-cost objective, as novelty.

Next Allowed:

- Expected-loss work needs a stronger structural estimator/objective, not another per-prompt monotonic cost normalization.

Full Text:

```text
DISPROVEN: Rank-normalized expected answer loss rescues result-space expected-cost discovery.
Conclusion: Added `expected_answer_loss_cost_normalization=rank`, which replaces each prompt's forced-result NLLs with within-prompt ranks before the exact policy expectation. The full-grid Stage 0 diagnostic did not clear the gate: exact-vs-boundary result-proj cosine was only `0.049551` and upstream cosine was `0.002584`, weaker than an earlier contrastive-margin decoder sign flip that still failed Stage 1. Sampled PG remained aligned with the rank objective (`0.723444`/`0.719965`) but only barely aligned with the boundary target (`0.027483`/`0.016271`).
Do not repeat: Do not run rank-normalized expected answer-loss Stage 1, or rank/scale transforms of the same full-enum expected-cost objective, as novelty.
Next allowed test: Expected-loss work needs a stronger structural estimator/objective, not another per-prompt monotonic cost normalization.
Source: `aiAgentWorkHistory/phase7/2026-05-30-rank-normalized-expected-loss-gate.md`
```
