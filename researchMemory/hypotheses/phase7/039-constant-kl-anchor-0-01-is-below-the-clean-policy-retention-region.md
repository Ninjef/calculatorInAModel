# Constant KL anchor `0.01` is below the clean policy-retention region.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-threshold.md

Summary:

- Anchor `0.01` kept injection-zero near chance and final eval `0.7850/0.9375`, but final calc fell to `0.7625/0.6425` and anchor agreement to `0.8825/0.7050`.

Questions:

- What did we learn about Constant KL anchor `0.01` is below the clean policy-retention region?
- Has Constant KL anchor `0.01` is below the clean policy-retention region been tested?
- Should we repeat Constant KL anchor `0.01` is below the clean policy-retention region?
- What is the status of Constant KL anchor `0.01` is below the clean policy-retention region?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-threshold.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, anchor weight `0.01`, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Floored or gated schedules around the `0.1` region, selective unfreezing, or policy acquisition that reduces active anchoring needs.

Full Text:

```text
MIXED: Constant KL anchor `0.01` is below the clean policy-retention region.
Conclusion: Anchor `0.01` kept injection-zero near chance and final eval `0.7850/0.9375`, but final calc fell to `0.7625/0.6425` and anchor agreement to `0.8825/0.7050`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `0.01`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Floored or gated schedules around the `0.1` region, selective unfreezing, or policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-threshold.md`
```
