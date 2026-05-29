# Tiny anchoring is redundant when the policy backbone is frozen.

Kind: hypothesis_memory
Status: NO-GAIN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-tiny-anchor.md

Summary:

- `--freeze-calculator-policy-backbone` plus KL anchor `0.01` kept anchor agreement `1.0000/0.9975` and learned calc `0.8200/0.8000`, but final eval `0.7125/0.8600` was slightly below no-anchor backbone freeze.

Questions:

- What did we learn about Tiny anchoring is redundant when the policy backbone is frozen?
- Has Tiny anchoring is redundant when the policy backbone is frozen been tested?
- Should we repeat Tiny anchoring is redundant when the policy backbone is frozen?
- What is the status of Tiny anchoring is redundant when the policy backbone is frozen?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-tiny-anchor.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, `--freeze-calculator-policy-backbone`, result-policy anchor `0.01`, LR `3e-4`, 400-step unfreeze as novelty.

Next Allowed:

- Improve downstream/readout adaptation under stable policy, use answer-utility-aware retention, or improve source-policy acquisition; tiny action-policy anchoring is not the missing ingredient here.

Full Text:

```text
NO-GAIN: Tiny anchoring is redundant when the policy backbone is frozen.
Conclusion: `--freeze-calculator-policy-backbone` plus KL anchor `0.01` kept anchor agreement `1.0000/0.9975` and learned calc `0.8200/0.8000`, but final eval `0.7125/0.8600` was slightly below no-anchor backbone freeze.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-policy-backbone`, result-policy anchor `0.01`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Improve downstream/readout adaptation under stable policy, use answer-utility-aware retention, or improve source-policy acquisition; tiny action-policy anchoring is not the missing ingredient here.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-tiny-anchor.md`
```
