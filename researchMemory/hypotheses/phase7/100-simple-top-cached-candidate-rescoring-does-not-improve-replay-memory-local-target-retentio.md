# Simple top-cached-candidate rescoring does not improve replay-memory local-target retention.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md

Summary:

- Adding optional `_rN` cached-candidate rescoring to replay-memory branches showed no benefit for the best low-fresh branch. At 200 steps, `memory_policy_reweighted_t1_u2_m30_r2` exactly tied no-rescore `u2_m30` (`0.6025` exact calc / `0.6016` sampled normal) at double the forced-score cost (`4` vs `2` scores per step), while heavier rescoring was worse: `r4` reached `0.5300` calc / `0.5781` normal and `r8` reached `0.4675` / `0.4609`. The 800+200 `r2` retention gate also exactly tied no-rescore `u2_m30`: target `0.9000` calc / `0.8750` normal and retention `0.7850` calc / `0.7656` normal.

Questions:

- What did we learn about Simple top-cached-candidate rescoring does not improve replay-memory local-target retention?
- Has Simple top-cached-candidate rescoring does not improve replay-memory local-target retention been tested?
- Should we repeat Simple top-cached-candidate rescoring does not improve replay-memory local-target retention?
- What is the status of Simple top-cached-candidate rescoring does not improve replay-memory local-target retention?
- Why did Simple top-cached-candidate rescoring does not improve replay-memory local-target retention fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 `u2_m30` rescore sweep over `r2/r4/r8` or the same `u2_m30_r2` 800+200 retention gate as novelty.

Next Allowed:

- Stop simple rescore-count tweaking. Attack transduction directly with finite/reset memory, streaming/non-exhaustive prompts, or learned/generalized candidate memory.

Full Text:

```text
MIXED-NEGATIVE: Simple top-cached-candidate rescoring does not improve replay-memory local-target retention.
Conclusion: Adding optional `_rN` cached-candidate rescoring to replay-memory branches showed no benefit for the best low-fresh branch. At 200 steps, `memory_policy_reweighted_t1_u2_m30_r2` exactly tied no-rescore `u2_m30` (`0.6025` exact calc / `0.6016` sampled normal) at double the forced-score cost (`4` vs `2` scores per step), while heavier rescoring was worse: `r4` reached `0.5300` calc / `0.5781` normal and `r8` reached `0.4675` / `0.4609`. The 800+200 `r2` retention gate also exactly tied no-rescore `u2_m30`: target `0.9000` calc / `0.8750` normal and retention `0.7850` calc / `0.7656` normal.
Do not repeat: The same seed-2 `u2_m30` rescore sweep over `r2/r4/r8` or the same `u2_m30_r2` 800+200 retention gate as novelty.
Next allowed test: Stop simple rescore-count tweaking. Attack transduction directly with finite/reset memory, streaming/non-exhaustive prompts, or learned/generalized candidate memory.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md`
```
