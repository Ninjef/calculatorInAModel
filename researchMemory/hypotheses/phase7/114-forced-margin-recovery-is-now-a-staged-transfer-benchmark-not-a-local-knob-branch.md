# Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-forced-margin-recovery-review.md

Summary:

- The forced-margin branch answered its post-review questions. Manual low-LR recovery raised one-negative forced-margin handoff to `0.8700` final / `0.9050` step-600 normal, and automated fresh-seed recovery raised source calc `0.5825 -> 0.8825` during the late window and reached `0.9875` trusted frozen-policy handoff final eval / `0.9800` step-600 normal. This makes automated one-negative forced-margin recovery the current best staged-transfer source recipe and a benchmark for future objectives, but not the final solution because it still depends on hard improvement assignment, true-result forced-margin pressure, and frozen-policy staged transfer.

Questions:

- What did we learn about Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch?
- Has Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch been tested?
- Should we repeat Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch?
- What is the status of Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch?
- Why did Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch fail?

Representative evidence:

- `researchReviews/2026-05-30-forced-margin-recovery-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun seed-15 manual recovery, seed-16 automated recovery plus handoff, or same-setup forced-margin start-step/margin/negative-count/late recovery length tweaks as novelty.

Next Allowed:

- Use the recipe as a benchmark; future compute should either stress scale/stability or remove prescriptiveness by replacing hard assignment or true-result forcing with a new target construction or estimator.

Full Text:

```text
REVIEW: Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch.
Conclusion: The forced-margin branch answered its post-review questions. Manual low-LR recovery raised one-negative forced-margin handoff to `0.8700` final / `0.9050` step-600 normal, and automated fresh-seed recovery raised source calc `0.5825 -> 0.8825` during the late window and reached `0.9875` trusted frozen-policy handoff final eval / `0.9800` step-600 normal. This makes automated one-negative forced-margin recovery the current best staged-transfer source recipe and a benchmark for future objectives, but not the final solution because it still depends on hard improvement assignment, true-result forced-margin pressure, and frozen-policy staged transfer.
Do not repeat: Do not rerun seed-15 manual recovery, seed-16 automated recovery plus handoff, or same-setup forced-margin start-step/margin/negative-count/late recovery length tweaks as novelty.
Next allowed test: Use the recipe as a benchmark; future compute should either stress scale/stability or remove prescriptiveness by replacing hard assignment or true-result forcing with a new target construction or estimator.
Source: `researchReviews/2026-05-30-forced-margin-recovery-review.md`
```
