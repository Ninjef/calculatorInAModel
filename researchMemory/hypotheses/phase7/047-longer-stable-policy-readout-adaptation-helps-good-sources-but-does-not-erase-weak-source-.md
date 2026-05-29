# Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-long-adaptation.md

Summary:

- `--freeze-calculator-policy-backbone`, no anchor, 1600 steps lifted `src5_add5` to final eval `0.9500` with learned calc `0.8325`, but `src4_add2` reached only `0.7550` despite learned calc `0.8550`.

Questions:

- What did we learn about Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity?
- Has Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity been tested?
- Should we repeat Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity?
- What is the status of Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-long-adaptation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 1600-step adaptation as novelty.

Next Allowed:

- Source-policy acquisition/selection, stronger downstream adaptation targeted at weak sources, or a utility-aware readout objective under stable policy.

Full Text:

```text
MIXED: Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity.
Conclusion: `--freeze-calculator-policy-backbone`, no anchor, 1600 steps lifted `src5_add5` to final eval `0.9500` with learned calc `0.8325`, but `src4_add2` reached only `0.7550` despite learned calc `0.8550`.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 1600-step adaptation as novelty.
Next allowed test: Source-policy acquisition/selection, stronger downstream adaptation targeted at weak sources, or a utility-aware readout objective under stable policy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-long-adaptation.md`
```
