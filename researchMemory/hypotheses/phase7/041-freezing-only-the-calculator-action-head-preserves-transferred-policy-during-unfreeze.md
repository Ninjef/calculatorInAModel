# Freezing only the calculator action head preserves transferred policy during unfreeze.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-freeze-action-head.md

Summary:

- With `result_proj` frozen and only upstream trainable, adapted `src4_add2/src5_add5` still collapsed to final calc `0.3000/0.2525` and final eval `0.5200/0.8100`, matching the earlier plain unfreeze failure.

Questions:

- What did we learn about Freezing only the calculator action head preserves transferred policy during unfreeze?
- Has Freezing only the calculator action head preserves transferred policy during unfreeze been tested?
- Should we repeat Freezing only the calculator action head preserves transferred policy during unfreeze?
- What is the status of Freezing only the calculator action head preserves transferred policy during unfreeze?
- Why did Freezing only the calculator action head preserves transferred policy during unfreeze fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-freeze-action-head.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, `--freeze-calculator-action-head`, no anchor, LR `3e-4`, 400-step unfreeze as novelty.

Next Allowed:

- Behavior-level anchoring/gating, freezing the upstream policy path, or a more targeted selective parameter set that prevents upstream representation drift.

Full Text:

```text
DISPROVEN: Freezing only the calculator action head preserves transferred policy during unfreeze.
Conclusion: With `result_proj` frozen and only upstream trainable, adapted `src4_add2/src5_add5` still collapsed to final calc `0.3000/0.2525` and final eval `0.5200/0.8100`, matching the earlier plain unfreeze failure.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-action-head`, no anchor, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Behavior-level anchoring/gating, freezing the upstream policy path, or a more targeted selective parameter set that prevents upstream representation drift.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-freeze-action-head.md`
```
