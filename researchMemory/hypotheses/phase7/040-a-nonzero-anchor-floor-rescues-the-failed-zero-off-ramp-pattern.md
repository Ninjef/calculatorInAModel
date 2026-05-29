# A nonzero anchor floor rescues the failed zero-off-ramp pattern.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-floor-schedule.md

Summary:

- Anchor `1.0 -> 0.1` over 200 steps kept final calc `0.8175/0.7800`, final eval `0.7925/0.9775`, and injection-zero `0.0250/0.0075`, but did not beat constant anchor `0.1`.

Questions:

- What did we learn about A nonzero anchor floor rescues the failed zero-off-ramp pattern?
- Has A nonzero anchor floor rescues the failed zero-off-ramp pattern been tested?
- Should we repeat A nonzero anchor floor rescues the failed zero-off-ramp pattern?
- What is the status of A nonzero anchor floor rescues the failed zero-off-ramp pattern?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-floor-schedule.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, anchor `1.0`, decay `200`, floor `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Calculator-accuracy-gated retention, adaptive floors, selective unfreeze, or source-policy acquisition that reduces active anchoring needs.

Full Text:

```text
PARTIAL: A nonzero anchor floor rescues the failed zero-off-ramp pattern.
Conclusion: Anchor `1.0 -> 0.1` over 200 steps kept final calc `0.8175/0.7800`, final eval `0.7925/0.9775`, and injection-zero `0.0250/0.0075`, but did not beat constant anchor `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor `1.0`, decay `200`, floor `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Calculator-accuracy-gated retention, adaptive floors, selective unfreeze, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-floor-schedule.md`
```
