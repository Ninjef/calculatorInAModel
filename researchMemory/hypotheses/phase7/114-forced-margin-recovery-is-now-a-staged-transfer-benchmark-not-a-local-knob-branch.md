# Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch.

Status: REVIEW / STEERING.

Source: researchReviews/2026-05-30-forced-margin-recovery-review.md

Summary:

- The forced-margin branch has now completed its two allowed post-review
  follow-ups: manual source recovery and fresh-seed automated recovery.
- Manual recovery raised the longer one-negative forced-margin source handoff
  to `0.8700` final / `0.9050` step-600 normal.
- Automated recovery on a fresh seed improved source calc from `0.5825` to
  `0.8825` during the late `600->630` window and reached `0.9875` trusted
  frozen-policy handoff final eval / `0.9800` step-600 normal.
- This makes automated one-negative forced-margin recovery the current best
  staged-transfer source recipe and a benchmark for future source objectives.
- It is not the final goal because it still uses hard improvement assignment,
  true-result forced-margin pressure, and frozen-policy staged transfer.

Questions this memory answers:

- Should we keep tuning forced-margin knobs?
- What did the forced-margin recovery review decide?
- Is automated forced-margin recovery a final solution?
- What forced-margin experiments should future agents avoid?
- What should forced-margin evidence be used for next?

Do not repeat:

- Do not rerun seed-15 manual recovery, seed-16 automated recovery plus
  handoff, or same-setup forced-margin start-step/margin/negative-count/late
  recovery length tweaks as novelty.

Next allowed test:

- Use automated forced-margin recovery as a benchmark. Future compute should
  either stress scale/stability or remove prescriptiveness by replacing hard
  assignment or true-result forcing with a new target construction or estimator.

Ledger entry:

REVIEW: Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch. Conclusion: The forced-margin branch answered its post-review questions. Manual low-LR recovery raised one-negative forced-margin handoff to `0.8700` final / `0.9050` step-600 normal, and automated fresh-seed recovery raised source calc `0.5825 -> 0.8825` during the late window and reached `0.9875` trusted frozen-policy handoff final eval / `0.9800` step-600 normal. This makes automated one-negative forced-margin recovery the current best staged-transfer source recipe and a benchmark for future objectives, but not the final solution because it still depends on hard improvement assignment, true-result forced-margin pressure, and frozen-policy staged transfer.
Do not repeat: Do not rerun seed-15 manual recovery, seed-16 automated recovery plus handoff, or same-setup forced-margin start-step/margin/negative-count/late recovery length tweaks as novelty.
Next allowed test: Use the recipe as a benchmark; future compute should either stress scale/stability or remove prescriptiveness by replacing hard assignment or true-result forcing with a new target construction or estimator.
Source: `researchReviews/2026-05-30-forced-margin-recovery-review.md`
