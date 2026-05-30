# Automated forced-margin recovery does not clear the op29 range stress.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-forced-margin-op29-range-stress.md

Summary:

- A matching wider product oracle decoder for `operand_max=29` reached full-grid `1.0000`, so decoder wiring was not the bottleneck. With the same automated one-negative forced-margin recovery recipe, the op29 source improved during late recovery from `0.3533` at step `600` to `0.6889` at step `630`, with final source eval `0.7133`. The trusted 600-step frozen-policy additive handoff reached `0.8533` final eval / `0.8278` step-600 normal, with low controls (`0.0344` injection-zero, `0.0189` forced-random at step `600`) but learned calc only `0.6522` at step `600`. This keeps the calculator path causal but fails the high non-bottleneck gate, showing that the op19 full-grid hard-assignment forced-margin recipe is not yet a scalable range solution.

Questions:

- What did we learn about Automated forced-margin recovery does not clear the op29 range stress?
- Has Automated forced-margin recovery does not clear the op29 range stress been tested?
- Should we repeat Automated forced-margin recovery does not clear the op29 range stress?
- What is the status of Automated forced-margin recovery does not clear the op29 range stress?
- Why did Automated forced-margin recovery does not clear the op29 range stress fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-op29-range-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same `operand_max=29`, product-decoder, effective-seed-29 source-plus-handoff, and do not jump to op49 with the identical full-grid hard-assignment recipe as novelty.

Next Allowed:

- Further range work should change source acquisition, reduce assignment cost with a predeclared exact-grid ceiling comparison, or test materially more source capacity/recovery only if the goal is to diagnose the op29 range failure mode.

Full Text:

```text
MIXED-NEGATIVE: Automated forced-margin recovery does not clear the op29 range stress.
Conclusion: A matching wider product oracle decoder for `operand_max=29` reached full-grid `1.0000`, so decoder wiring was not the bottleneck. With the same automated one-negative forced-margin recovery recipe, the op29 source improved during late recovery from `0.3533` at step `600` to `0.6889` at step `630`, with final source eval `0.7133`. The trusted 600-step frozen-policy additive handoff reached `0.8533` final eval / `0.8278` step-600 normal, with low controls (`0.0344` injection-zero, `0.0189` forced-random at step `600`) but learned calc only `0.6522` at step `600`. This keeps the calculator path causal but fails the high non-bottleneck gate, showing that the op19 full-grid hard-assignment forced-margin recipe is not yet a scalable range solution.
Do not repeat: Do not rerun the same `operand_max=29`, product-decoder, effective-seed-29 source-plus-handoff, and do not jump to op49 with the identical full-grid hard-assignment recipe as novelty.
Next allowed test: Further range work should change source acquisition, reduce assignment cost with a predeclared exact-grid ceiling comparison, or test materially more source capacity/recovery only if the goal is to diagnose the op29 range failure mode.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-op29-range-stress.md`
```
