# Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md

Summary:

- Adding `memory_policy_reweighted_t1_u8_m24` to the Stage 1 local-target runner lets each prompt cache observed forced-result losses and train on 8 fresh uniform candidates plus 24 low-loss cached candidates. At 200 steps it beat the raw uniform `u32` baseline while scoring one quarter as many fresh results per step: exact-grid calc `0.5900` and sampled normal `0.5391` versus `0.3350`/`0.3438`; target true-candidate coverage reached `1.0000`, target argmax accuracy `0.9850`, and controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, the memory branch reached target `0.9600` exact calc / `0.9766` sampled normal and retained `0.8600` calc / `0.8750` sampled normal under answer-only training.

Questions:

- What did we learn about Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step?
- Has Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step been tested?
- Should we repeat Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step?
- What is the status of Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step?
- What follow-up is allowed for Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 replay-memory `u8_m24` versus raw uniform `u32` 200-step gate or the same single-branch 800+200 retention gate as novelty.

Next Allowed:

- Stress scalability rather than rerun the positive: reduce fresh scoring (`u4` or lower), add aging/rescoring to handle stale losses, or test whether a learned/generalized memory proposal works beyond the fixed exhaustive grid.

Full Text:

```text
POSITIVE: Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step.
Conclusion: Adding `memory_policy_reweighted_t1_u8_m24` to the Stage 1 local-target runner lets each prompt cache observed forced-result losses and train on 8 fresh uniform candidates plus 24 low-loss cached candidates. At 200 steps it beat the raw uniform `u32` baseline while scoring one quarter as many fresh results per step: exact-grid calc `0.5900` and sampled normal `0.5391` versus `0.3350`/`0.3438`; target true-candidate coverage reached `1.0000`, target argmax accuracy `0.9850`, and controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, the memory branch reached target `0.9600` exact calc / `0.9766` sampled normal and retained `0.8600` calc / `0.8750` sampled normal under answer-only training.
Do not repeat: The same seed-2 replay-memory `u8_m24` versus raw uniform `u32` 200-step gate or the same single-branch 800+200 retention gate as novelty.
Next allowed test: Stress scalability rather than rerun the positive: reduce fresh scoring (`u4` or lower), add aging/rescoring to handle stale losses, or test whether a learned/generalized memory proposal works beyond the fixed exhaustive grid.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md`
```
