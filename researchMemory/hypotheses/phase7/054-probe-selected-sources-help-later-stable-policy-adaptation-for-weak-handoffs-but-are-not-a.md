# Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-probe-selected-policy-backbone-adaptation.md

Summary:

- 1600-step no-anchor `--freeze-calculator-policy-backbone` adaptation from probe-selected checkpoints lifted `src4` from old final-source long adaptation `0.7550` to `0.8900`, but `src5` reached `0.9250`, below the old final-source long adaptation `0.9500`.

Questions:

- What did we learn about Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector?
- Has Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector been tested?
- Should we repeat Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector?
- What is the status of Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector?
- What follow-up is allowed for Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-probe-selected-policy-backbone-adaptation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same probe-selected `src4` step-1200/add2 or `src5` step-1100/add5 frozen handoff checkpoint into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.

Next Allowed:

- Add a second-stage long-adaptation/readout-compatibility selector, optimize source acquisition for both 600-step handoff slope and later readout adaptability, or reduce the handoff-probe cost.

Full Text:

```text
MIXED-POSITIVE: Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector.
Conclusion: 1600-step no-anchor `--freeze-calculator-policy-backbone` adaptation from probe-selected checkpoints lifted `src4` from old final-source long adaptation `0.7550` to `0.8900`, but `src5` reached `0.9250`, below the old final-source long adaptation `0.9500`.
Do not repeat: Same probe-selected `src4` step-1200/add2 or `src5` step-1100/add5 frozen handoff checkpoint into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Add a second-stage long-adaptation/readout-compatibility selector, optimize source acquisition for both 600-step handoff slope and later readout adaptability, or reduce the handoff-probe cost.
Source: `aiAgentWorkHistory/phase7/2026-05-29-probe-selected-policy-backbone-adaptation.md`
```
