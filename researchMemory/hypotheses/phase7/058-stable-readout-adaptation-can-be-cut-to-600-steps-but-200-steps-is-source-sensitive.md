# Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-reduced-readout-budget-validation.md

Summary:

- From continued selected checkpoints, 200-step readout worked for `src5` (`0.9275`) but not `src4` (`0.8775`); 600-step readout passed both (`src4 0.9025`, `src5 0.9325`) with injection-zero near zero and forced-random near chance.

Questions:

- What did we learn about Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive?
- Has Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive been tested?
- Should we repeat Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive?
- What is the status of Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive?
- What follow-up is allowed for Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-reduced-readout-budget-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same 200/600-step no-anchor policy-backbone-frozen readout adaptation from continued selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.

Next Allowed:

- Reduce the 600-step handoff probe or 800-step frozen-policy continuation cost, or optimize source acquisition for early handoff/continuation slope.

Full Text:

```text
MIXED-POSITIVE: Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive.
Conclusion: From continued selected checkpoints, 200-step readout worked for `src5` (`0.9275`) but not `src4` (`0.8775`); 600-step readout passed both (`src4 0.9025`, `src5 0.9325`) with injection-zero near zero and forced-random near chance.
Do not repeat: Same 200/600-step no-anchor policy-backbone-frozen readout adaptation from continued selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Reduce the 600-step handoff probe or 800-step frozen-policy continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-readout-budget-validation.md`
```
