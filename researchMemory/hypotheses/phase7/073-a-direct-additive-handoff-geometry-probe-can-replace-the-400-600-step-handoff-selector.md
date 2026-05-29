# A direct additive handoff geometry probe can replace the 400/600-step handoff selector.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-handoff-geometry-probe.md

Summary:

- Forced-result geometry flags seed-10 as hostile (`true_best=0.0`, true top-3 `0.03-0.045`, true-best gap `0.0058-0.0063`) versus seed-9 positive (`true_best=0.0625`, top-3 `0.2125`, gap `0.0034`), but it does not cleanly separate `src6` positive from `src7` boundary-negative and 100-step loss slope is not a reliable selector.

Questions:

- What did we learn about A direct additive handoff geometry probe can replace the 400/600-step handoff selector?
- Has A direct additive handoff geometry probe can replace the 400/600-step handoff selector been tested?
- Should we repeat A direct additive handoff geometry probe can replace the 400/600-step handoff selector?
- What is the status of A direct additive handoff geometry probe can replace the 400/600-step handoff selector?
- Why did A direct additive handoff geometry probe can replace the 400/600-step handoff selector fail?
- What follow-up is allowed for A direct additive handoff geometry probe can replace the 400/600-step handoff selector?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-handoff-geometry-probe.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same geometry probe over seed-9 final, seed-10 `1000/1300/1400/final`, `src6` final, or `src7` step `1400` as novelty.

Next Allowed:

- Add forced-result geometry as a source-training snapshot metric, optimize it during source acquisition, or keep using actual handoff probes as selection gates.

Full Text:

```text
MIXED-NEGATIVE: A direct additive handoff geometry probe can replace the 400/600-step handoff selector.
Conclusion: Forced-result geometry flags seed-10 as hostile (`true_best=0.0`, true top-3 `0.03-0.045`, true-best gap `0.0058-0.0063`) versus seed-9 positive (`true_best=0.0625`, top-3 `0.2125`, gap `0.0034`), but it does not cleanly separate `src6` positive from `src7` boundary-negative and 100-step loss slope is not a reliable selector.
Do not repeat: Same geometry probe over seed-9 final, seed-10 `1000/1300/1400/final`, `src6` final, or `src7` step `1400` as novelty.
Next allowed test: Add forced-result geometry as a source-training snapshot metric, optimize it during source acquisition, or keep using actual handoff probes as selection gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-handoff-geometry-probe.md`
```
