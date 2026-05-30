# Automated forced-margin recovery survives a wider model scale stress.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-forced-margin-wider-model-scale-stress.md

Summary:

- Using an existing wider semantic decoder (`n_embd=32`, `n_head=2`, non-product answer decoder), the automated one-negative forced-margin recovery recipe remained viable and transferred strongly. The wider source reached `0.9125` final eval and improved source calc from `0.7825` at step `600` to `0.8825` at step `630`. The trusted frozen-policy additive handoff from step `630` reached `1.0000` final eval and `1.0000` step-600 normal, with zero-injection `0.0625`, forced-random `0.0325`, and learned calc `0.8850` at step `600`. This supports scale/stability of the staged benchmark, but it is still prescriptive and has a non-product decoder caveat.

Questions:

- What did we learn about Automated forced-margin recovery survives a wider model scale stress?
- Has Automated forced-margin recovery survives a wider model scale stress been tested?
- Should we repeat Automated forced-margin recovery survives a wider model scale stress?
- What is the status of Automated forced-margin recovery survives a wider model scale stress?
- Why did Automated forced-margin recovery survives a wider model scale stress fail?
- What follow-up is allowed for Automated forced-margin recovery survives a wider model scale stress?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-wider-model-scale-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same `n_embd=32`, `n_head=2`, effective-seed-25 wider forced-margin source plus 600-step handoff as novelty.

Next Allowed:

- Further scale work should use a matching product semantic decoder, larger operand range, larger architecture family, or remove hard assignment / true-result forcing rather than tuning local forced-margin knobs.

Full Text:

```text
POSITIVE: Automated forced-margin recovery survives a wider model scale stress.
Conclusion: Using an existing wider semantic decoder (`n_embd=32`, `n_head=2`, non-product answer decoder), the automated one-negative forced-margin recovery recipe remained viable and transferred strongly. The wider source reached `0.9125` final eval and improved source calc from `0.7825` at step `600` to `0.8825` at step `630`. The trusted frozen-policy additive handoff from step `630` reached `1.0000` final eval and `1.0000` step-600 normal, with zero-injection `0.0625`, forced-random `0.0325`, and learned calc `0.8850` at step `600`. This supports scale/stability of the staged benchmark, but it is still prescriptive and has a non-product decoder caveat.
Do not repeat: Do not rerun the same `n_embd=32`, `n_head=2`, effective-seed-25 wider forced-margin source plus 600-step handoff as novelty.
Next allowed test: Further scale work should use a matching product semantic decoder, larger operand range, larger architecture family, or remove hard assignment / true-result forcing rather than tuning local forced-margin knobs.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-wider-model-scale-stress.md`
```
