# Automated forced-margin recovery replicates strongly on a fresh seed.

Status: POSITIVE, constrained.

Source: aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md

Summary:

- Added `--late-source-recovery-additive-forced-margin-loss-weight` so the
  late source-recovery phase can reduce forced-margin pressure in the same
  source run instead of requiring a manual checkpoint continuation.
- On a fresh forced-margin source seed, the automated 630-step run activated
  recovery at step `600` with LR multiplier `0.1` and forced-margin weight
  `0.1`.
- Source calculator accuracy jumped from `0.5825` at step `600` to `0.8825`
  at step `630`; final source eval was `0.8700`.
- The trusted frozen-policy 600-step non-bottleneck handoff from source step
  `630` reached `0.9875` final eval and `0.9800` step-600 normal, with
  injection-zero `0.0156-0.0250`, forced-random `0.0938`, and learned
  calculator accuracy `0.8906`.
- This is the strongest forced-margin handoff so far and shows the recovery
  recipe can be automated and survive a fresh seed. It remains prescriptive:
  source acquisition still uses hard improvement assignment plus true-result
  contrastive forcing.

Questions this memory answers:

- Can forced-margin low-LR recovery be automated in one source run?
- Does automated forced-margin recovery replicate on a fresh seed?
- What handoff accuracy did automated forced-margin recovery reach?
- Is automated forced-margin recovery the final scalable non-prescriptive method?
- What forced-margin automated recovery run should future agents avoid repeating?

Do not repeat:

- Do not rerun the same seed-16 630-step automated forced-margin recovery with
  late step `600`, LR multiplier `0.1`, margin weight override `0.1`, and the
  same 600-step frozen-policy handoff as novelty.

Next allowed test:

- If staying in forced-margin, test broader stability/scale or fold the idea
  into a less-prescriptive target construction. Otherwise pivot back to
  scalable credit assignment; this is strong evidence for staged transfer but
  does not solve non-prescriptive discovery.

Ledger entry:

POSITIVE: Automated forced-margin recovery replicates strongly on a fresh seed. Conclusion: Adding `--late-source-recovery-additive-forced-margin-loss-weight` folded the manual one-negative forced-margin recovery into a single source run. On fresh seed `16`, the step-600 late phase (`lr` multiplier `0.1`, forced-margin weight override `0.1`) improved source calc from `0.5825` at step `600` to `0.8825` at step `630`; final source eval was `0.8700`. The trusted frozen-policy 600-step non-bottleneck handoff from source step `630` reached `0.9875` final eval / `0.9800` step-600 normal, with injection-zero `0.0156-0.0250`, forced-random `0.0938`, and learned calc `0.8906`. This is the strongest forced-margin handoff so far and replicates the recovery mechanism beyond the manual checkpoint continuation, but it remains prescriptive because source training still uses hard assignment and true-result contrastive forcing.
Do not repeat: Do not rerun the same seed-16 630-step automated forced-margin recovery with late step `600`, LR multiplier `0.1`, margin weight override `0.1`, and the same 600-step frozen-policy handoff as novelty.
Next allowed test: If staying in forced-margin, test broader stability/scale or use it as a stepping stone toward less-prescriptive target construction; otherwise pivot back to scalable credit assignment, because this does not solve answer-loss-only discovery.
Source: `aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md`
