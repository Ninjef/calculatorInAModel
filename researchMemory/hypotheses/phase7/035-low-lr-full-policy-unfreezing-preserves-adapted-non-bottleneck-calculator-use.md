# Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-low-lr-unfreeze.md

Summary:

- From adapted weak-source checkpoints, unfreezing all policy parameters at LR `3e-4` for 400 steps collapsed learned calc from `0.8725 -> 0.3000` and `0.8000 -> 0.2525`; answer accuracy did not improve.

Questions:

- What did we learn about Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use?
- Has Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use been tested?
- Should we repeat Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use?
- What is the status of Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use?
- Why did Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-low-lr-unfreeze.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same `src4_add2` or `src5_add5` adapted-checkpoint low-LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Selective unfreezing, explicit policy-retention regularization, or unfreeze schedules gated by calculator-result accuracy.

Full Text:

```text
DISPROVEN: Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use.
Conclusion: From adapted weak-source checkpoints, unfreezing all policy parameters at LR `3e-4` for 400 steps collapsed learned calc from `0.8725 -> 0.3000` and `0.8000 -> 0.2525`; answer accuracy did not improve.
Do not repeat: The same `src4_add2` or `src5_add5` adapted-checkpoint low-LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Selective unfreezing, explicit policy-retention regularization, or unfreeze schedules gated by calculator-result accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-low-lr-unfreeze.md`
```
