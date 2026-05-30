# Four routed calculator hooks train and transfer under corrected controls.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-four-hook-routed-source-handoff.md

Summary:

- Stress-tested the corrected-control routed recipe from two hooks to four hooks using `left_operand_mod` routing, cloned output projections, `embd32`, and topk8+unique24 source assignment. The four-hook source630 reached `398/400 = 0.9950` final eval and step-630 normal/source-calc `0.9975`, with low controls (`0.0275` injection-zero, `0.0225` forced-random) and all hooks trained on the 400-sample snapshot (`0.9928/1.0000/1.0000/1.0000` hook calc). The trusted 600-step frozen-policy additive handoff from that source reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with corrected controls still low (`0.0400` injection-zero, `0.0200` forced-random) and all four hooks perfect on the final snapshot. This is the first more-than-two-hook routed non-bottleneck positive, directly advancing the many-calculator axis. Caveat: the current implementation still executes every hook before route masking and uses cloned per-hook output projections, so it proves trainability/transfer under route partitioning, not efficient active-only execution or parameter scaling.

Questions:

- What did we learn about Four routed calculator hooks train and transfer under corrected controls?
- Has Four routed calculator hooks train and transfer under corrected controls been tested?
- Should we repeat Four routed calculator hooks train and transfer under corrected controls?
- What is the status of Four routed calculator hooks train and transfer under corrected controls?
- What follow-up is allowed for Four routed calculator hooks train and transfer under corrected controls?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-four-hook-routed-source-handoff.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same effective-seed-43 op19 four-hook source630/handoff600 path as novelty. The 4-hook route-partition gate is positive.

Next Allowed:

- Implement active-only routed hook execution and/or shared/tied output projection, then validate the same 4-hook gate with compute/parameter accounting; alternatively run a fresh-seed 4-hook replication only if needed for stability after the efficiency change.

Full Text:

```text
POSITIVE: Four routed calculator hooks train and transfer under corrected controls.
Conclusion: Stress-tested the corrected-control routed recipe from two hooks to four hooks using `left_operand_mod` routing, cloned output projections, `embd32`, and topk8+unique24 source assignment. The four-hook source630 reached `398/400 = 0.9950` final eval and step-630 normal/source-calc `0.9975`, with low controls (`0.0275` injection-zero, `0.0225` forced-random) and all hooks trained on the 400-sample snapshot (`0.9928/1.0000/1.0000/1.0000` hook calc). The trusted 600-step frozen-policy additive handoff from that source reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with corrected controls still low (`0.0400` injection-zero, `0.0200` forced-random) and all four hooks perfect on the final snapshot. This is the first more-than-two-hook routed non-bottleneck positive, directly advancing the many-calculator axis. Caveat: the current implementation still executes every hook before route masking and uses cloned per-hook output projections, so it proves trainability/transfer under route partitioning, not efficient active-only execution or parameter scaling.
Do not repeat: Do not rerun the same effective-seed-43 op19 four-hook source630/handoff600 path as novelty. The 4-hook route-partition gate is positive.
Next allowed test: Implement active-only routed hook execution and/or shared/tied output projection, then validate the same 4-hook gate with compute/parameter accounting; alternatively run a fresh-seed 4-hook replication only if needed for stability after the efficiency change.
Source: `aiAgentWorkHistory/phase7/2026-05-30-four-hook-routed-source-handoff.md`
```
