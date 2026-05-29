# Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-freeze.md

Summary:

- `--freeze-calculator-policy-backbone` with no anchor preserved final learned calc `0.8200/0.8025` and improved adapted weak handoffs to final eval `0.7250/0.8700`, but stayed below lightweight anchor results.

Questions:

- What did we learn about Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing?
- Has Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing been tested?
- Should we repeat Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing?
- What is the status of Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-freeze.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 400-step unfreeze as novelty.

Next Allowed:

- Combine policy-backbone freezing with lightweight/utility-aware retention, improve source-policy acquisition, or test whether a different movable parameter set improves readout without policy drift.

Full Text:

```text
PARTIAL: Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing.
Conclusion: `--freeze-calculator-policy-backbone` with no anchor preserved final learned calc `0.8200/0.8025` and improved adapted weak handoffs to final eval `0.7250/0.8700`, but stayed below lightweight anchor results.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Combine policy-backbone freezing with lightweight/utility-aware retention, improve source-policy acquisition, or test whether a different movable parameter set improves readout without policy drift.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-freeze.md`
```
