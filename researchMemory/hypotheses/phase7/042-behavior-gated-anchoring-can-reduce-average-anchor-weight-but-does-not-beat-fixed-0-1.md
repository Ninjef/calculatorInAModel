# Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-behavior-gated-anchor.md

Summary:

- Base anchor `0.01` with agreement gate `<0.9 -> 0.1` ended with final calc `0.7700/0.7700` and final eval `0.8050/0.9675`; it improved over constant `0.01` but was roughly comparable to, not better than, constant `0.1`.

Questions:

- What did we learn about Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`?
- Has Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1` been tested?
- Should we repeat Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`?
- What is the status of Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-behavior-gated-anchor.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate threshold `0.9`, gate weight `0.1`, argmax-agreement gate, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Better gate metric/threshold, adaptive continuous weights, calculator-accuracy-gated retention, or source-policy acquisition that reduces active anchoring needs.

Full Text:

```text
PARTIAL: Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with agreement gate `<0.9 -> 0.1` ended with final calc `0.7700/0.7700` and final eval `0.8050/0.9675`; it improved over constant `0.01` but was roughly comparable to, not better than, constant `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate threshold `0.9`, gate weight `0.1`, argmax-agreement gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Better gate metric/threshold, adaptive continuous weights, calculator-accuracy-gated retention, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-behavior-gated-anchor.md`
```
