# One-negative forced-margin is a useful auxiliary, not a standalone new mainline.

Kind: hypothesis_memory
Status: CONSTRAINED
Phase: Phase 7
Source: researchReviews/2026-05-29-forced-margin-branch-review.md

Summary:

- Reviewing the forced-margin branch shows a real but bounded result. The one-negative objective is the scalable/practical variant and improves early full-grid handoff, but many-negative full-grid margin is too costly, longer same-seed one-negative training is checkpoint-sensitive, and the best longer handoff (`0.7400` final / `0.7850` snapshot) does not clearly surpass scheduled forced-true step-600 (`0.7725` final). The branch still relies on hard assignment and true-result forcing, so it does not solve non-prescriptive scalable credit assignment.

Questions:

- What did we learn about One-negative forced-margin is a useful auxiliary, not a standalone new mainline?
- Has One-negative forced-margin is a useful auxiliary, not a standalone new mainline been tested?
- Should we repeat One-negative forced-margin is a useful auxiliary, not a standalone new mainline?
- What is the status of One-negative forced-margin is a useful auxiliary, not a standalone new mainline?
- Why did One-negative forced-margin is a useful auxiliary, not a standalone new mainline fail?

Representative evidence:

- `researchReviews/2026-05-29-forced-margin-branch-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not continue with negative-count tweaks, same-seed longer ladders, start-step tweaks, slope-proxy selection, or geometry-only checkpoint fishing as novelty.

Next Allowed:

- Stay in forced-margin only for a predeclared source recovery/retention test or a fresh-seed stability replication with trusted 600-step handoff. Otherwise move effort toward learned/generalized proposals, estimator correction, or a less prescriptive target construction.

Full Text:

```text
CONSTRAINED: One-negative forced-margin is a useful auxiliary, not a standalone new mainline.
Conclusion: Reviewing the forced-margin branch shows a real but bounded result. The one-negative objective is the scalable/practical variant and improves early full-grid handoff, but many-negative full-grid margin is too costly, longer same-seed one-negative training is checkpoint-sensitive, and the best longer handoff (`0.7400` final / `0.7850` snapshot) does not clearly surpass scheduled forced-true step-600 (`0.7725` final). The branch still relies on hard assignment and true-result forcing, so it does not solve non-prescriptive scalable credit assignment.
Do not repeat: Do not continue with negative-count tweaks, same-seed longer ladders, start-step tweaks, slope-proxy selection, or geometry-only checkpoint fishing as novelty.
Next allowed test: Stay in forced-margin only for a predeclared source recovery/retention test or a fresh-seed stability replication with trusted 600-step handoff. Otherwise move effort toward learned/generalized proposals, estimator correction, or a less prescriptive target construction.
Source: `researchReviews/2026-05-29-forced-margin-branch-review.md`
```
