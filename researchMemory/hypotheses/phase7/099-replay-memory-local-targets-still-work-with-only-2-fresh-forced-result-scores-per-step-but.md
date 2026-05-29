# Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md

Summary:

- A lower fresh-score budget sweep compared raw uniform `u32` with replay-memory `u8_m24`, `u4_m28`, `u2_m30`, and `u1_m31`. At 200 steps, `u2_m30` was best: exact-grid calc `0.6025` and sampled normal `0.6016` with only 2 fresh forced-result scores per step, versus `u8_m24` `0.5900`/`0.5391`, `u4_m28` `0.5100`/`0.4844`, `u1_m31` `0.4075`/`0.4219`, and raw `u32` `0.3350`/`0.3438`; controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, `u2_m30` reached target `0.9000` calc / `0.8750` normal and retained `0.7850` calc / `0.7656` normal, below the prior `u8_m24` retention (`0.8600`/`0.8750`) but still far above sparse uniform baselines.

Questions:

- What did we learn about Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens?
- Has Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens been tested?
- Should we repeat Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens?
- What is the status of Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens?
- What follow-up is allowed for Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 lower-budget sweep over `u8_m24/u4_m28/u2_m30/u1_m31` or the same `u2_m30` 800+200 retention gate as novelty.

Next Allowed:

- Move from budget sweeps to scalability stressors: stale-cache aging/rescoring, memory reset, streaming/non-exhaustive prompts, or learned/generalized candidate memory. Treat `u2_m30` as the best current low-fresh-score point and `u1_m31` as below the useful budget floor at 200 steps.

Full Text:

```text
POSITIVE: Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens.
Conclusion: A lower fresh-score budget sweep compared raw uniform `u32` with replay-memory `u8_m24`, `u4_m28`, `u2_m30`, and `u1_m31`. At 200 steps, `u2_m30` was best: exact-grid calc `0.6025` and sampled normal `0.6016` with only 2 fresh forced-result scores per step, versus `u8_m24` `0.5900`/`0.5391`, `u4_m28` `0.5100`/`0.4844`, `u1_m31` `0.4075`/`0.4219`, and raw `u32` `0.3350`/`0.3438`; controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, `u2_m30` reached target `0.9000` calc / `0.8750` normal and retained `0.7850` calc / `0.7656` normal, below the prior `u8_m24` retention (`0.8600`/`0.8750`) but still far above sparse uniform baselines.
Do not repeat: The same seed-2 lower-budget sweep over `u8_m24/u4_m28/u2_m30/u1_m31` or the same `u2_m30` 800+200 retention gate as novelty.
Next allowed test: Move from budget sweeps to scalability stressors: stale-cache aging/rescoring, memory reset, streaming/non-exhaustive prompts, or learned/generalized candidate memory. Treat `u2_m30` as the best current low-fresh-score point and `u1_m31` as below the useful budget floor at 200 steps.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md`
```
