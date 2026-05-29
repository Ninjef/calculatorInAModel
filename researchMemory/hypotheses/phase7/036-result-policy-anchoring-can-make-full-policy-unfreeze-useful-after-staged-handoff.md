# Result-policy anchoring can make full-policy unfreeze useful after staged handoff.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-policy-anchor-unfreeze.md

Summary:

- KL anchor weight `10` at LR `3e-4` preserved learned calc (`0.8075/0.7950`) and improved final eval over frozen adapted baselines (`src4_add2 0.6050 -> 0.7475`, `src5_add5 0.8175 -> 0.9525`).

Questions:

- What did we learn about Result-policy anchoring can make full-policy unfreeze useful after staged handoff?
- Has Result-policy anchoring can make full-policy unfreeze useful after staged handoff been tested?
- Should we repeat Result-policy anchoring can make full-policy unfreeze useful after staged handoff?
- What is the status of Result-policy anchoring can make full-policy unfreeze useful after staged handoff?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-policy-anchor-unfreeze.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, anchor weight `10`, LR `3e-4`, 400-step KL-anchor full unfreeze as novelty.

Next Allowed:

- Anchor decay/off-ramp, selective unfreeze, source checkpoint selection, or less prescriptive source-policy acquisition.

Full Text:

```text
PARTIAL: Result-policy anchoring can make full-policy unfreeze useful after staged handoff.
Conclusion: KL anchor weight `10` at LR `3e-4` preserved learned calc (`0.8075/0.7950`) and improved final eval over frozen adapted baselines (`src4_add2 0.6050 -> 0.7475`, `src5_add5 0.8175 -> 0.9525`).
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, LR `3e-4`, 400-step KL-anchor full unfreeze as novelty.
Next allowed test: Anchor decay/off-ramp, selective unfreeze, source checkpoint selection, or less prescriptive source-policy acquisition.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-policy-anchor-unfreeze.md`
```
