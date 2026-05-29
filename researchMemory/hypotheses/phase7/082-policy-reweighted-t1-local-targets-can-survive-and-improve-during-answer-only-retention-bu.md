# `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md

Summary:

- In an 800-step target-training plus 200-step answer-only retention gate, `policy_reweighted_t1` trailed hard-boundary at target step 800 (`0.7050` vs `0.8200` exact-grid calc) after peaking at step 600 (`0.8925`), but finished retention at `0.8925` exact-grid calc and `0.8750` sampled normal versus hard-boundary `0.8050`/`0.8281`. Controls remained causal (`injection_zero=0.0234`, `forced_random=0.0156`, oracle `1.0000`).

Questions:

- What did we learn about `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic?
- Has `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic been tested?
- Should we repeat `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic?
- What is the status of `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic?
- What follow-up is allowed for `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2, 800-target-step plus 200-retention-step comparison of `hard_boundary` and `policy_reweighted_t1` as novelty.

Next Allowed:

- Seed-replicate only if stability is the explicit question; otherwise approximate `policy_reweighted_t1` with sampled/top-k/learned targets that avoid full forced-result enumeration.

Full Text:

```text
PARTIAL-POSITIVE: `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic.
Conclusion: In an 800-step target-training plus 200-step answer-only retention gate, `policy_reweighted_t1` trailed hard-boundary at target step 800 (`0.7050` vs `0.8200` exact-grid calc) after peaking at step 600 (`0.8925`), but finished retention at `0.8925` exact-grid calc and `0.8750` sampled normal versus hard-boundary `0.8050`/`0.8281`. Controls remained causal (`injection_zero=0.0234`, `forced_random=0.0156`, oracle `1.0000`).
Do not repeat: The same seed-2, 800-target-step plus 200-retention-step comparison of `hard_boundary` and `policy_reweighted_t1` as novelty.
Next allowed test: Seed-replicate only if stability is the explicit question; otherwise approximate `policy_reweighted_t1` with sampled/top-k/learned targets that avoid full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`
```
