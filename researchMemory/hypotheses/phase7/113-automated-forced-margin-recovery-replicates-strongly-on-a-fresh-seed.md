# Automated forced-margin recovery replicates strongly on a fresh seed.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md

Summary:

- Adding `--late-source-recovery-additive-forced-margin-loss-weight` folded the manual one-negative forced-margin recovery into a single source run. On fresh seed `16`, the step-600 late phase (`lr` multiplier `0.1`, forced-margin weight override `0.1`) improved source calc from `0.5825` at step `600` to `0.8825` at step `630`; final source eval was `0.8700`. The trusted frozen-policy 600-step non-bottleneck handoff from source step `630` reached `0.9875` final eval / `0.9800` step-600 normal, with injection-zero `0.0156-0.0250`, forced-random `0.0938`, and learned calc `0.8906`. This is the strongest forced-margin handoff so far and replicates the recovery mechanism beyond the manual checkpoint continuation, but it remains prescriptive because source training still uses hard assignment and true-result contrastive forcing.

Questions:

- What did we learn about Automated forced-margin recovery replicates strongly on a fresh seed?
- Has Automated forced-margin recovery replicates strongly on a fresh seed been tested?
- Should we repeat Automated forced-margin recovery replicates strongly on a fresh seed?
- What is the status of Automated forced-margin recovery replicates strongly on a fresh seed?
- Why did Automated forced-margin recovery replicates strongly on a fresh seed fail?
- What follow-up is allowed for Automated forced-margin recovery replicates strongly on a fresh seed?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same seed-16 630-step automated forced-margin recovery with late step `600`, LR multiplier `0.1`, margin weight override `0.1`, and the same 600-step frozen-policy handoff as novelty.

Next Allowed:

- If staying in forced-margin, test broader stability/scale or use it as a stepping stone toward less-prescriptive target construction; otherwise pivot back to scalable credit assignment, because this does not solve answer-loss-only discovery.

Full Text:

```text
POSITIVE: Automated forced-margin recovery replicates strongly on a fresh seed.
Conclusion: Adding `--late-source-recovery-additive-forced-margin-loss-weight` folded the manual one-negative forced-margin recovery into a single source run. On fresh seed `16`, the step-600 late phase (`lr` multiplier `0.1`, forced-margin weight override `0.1`) improved source calc from `0.5825` at step `600` to `0.8825` at step `630`; final source eval was `0.8700`. The trusted frozen-policy 600-step non-bottleneck handoff from source step `630` reached `0.9875` final eval / `0.9800` step-600 normal, with injection-zero `0.0156-0.0250`, forced-random `0.0938`, and learned calc `0.8906`. This is the strongest forced-margin handoff so far and replicates the recovery mechanism beyond the manual checkpoint continuation, but it remains prescriptive because source training still uses hard assignment and true-result contrastive forcing.
Do not repeat: Do not rerun the same seed-16 630-step automated forced-margin recovery with late step `600`, LR multiplier `0.1`, margin weight override `0.1`, and the same 600-step frozen-policy handoff as novelty.
Next allowed test: If staying in forced-margin, test broader stability/scale or use it as a stepping stone toward less-prescriptive target construction; otherwise pivot back to scalable credit assignment, because this does not solve answer-loss-only discovery.
Source: `aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md`
```
