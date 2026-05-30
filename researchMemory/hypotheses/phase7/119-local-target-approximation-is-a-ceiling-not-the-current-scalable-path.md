# Local-target approximation is a ceiling, not the current scalable path.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-local-target-approximation-direction-review.md

Summary:

- Exact `policy_reweighted_t1` remains a useful proof of principle and diagnostic ceiling, but the tested scalable approximation families have now failed from enough angles to pause the branch as a mainline. Sparse uniform/top-k and adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, imputed sparse targets dilute pressure, simple learned proposals do not retain lift under streaming, random-prompt warmup is mixed-negative, and sparse pairwise preference failed even when `u32` covered the true result in `0.8450` of prompts (`0.0425` exact calc / `0.0234` sampled normal versus same-budget policy-reweighted `u32` at `0.3350` / `0.3438`).

Questions:

- What did we learn about Local-target approximation is a ceiling, not the current scalable path?
- Has Local-target approximation is a ceiling, not the current scalable path been tested?
- Should we repeat Local-target approximation is a ceiling, not the current scalable path?
- What is the status of Local-target approximation is a ceiling, not the current scalable path?
- Why did Local-target approximation is a ceiling, not the current scalable path fail?

Representative evidence:

- `researchReviews/2026-05-30-local-target-approximation-direction-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more sparse count ladders, top-k/neighborhood proposal tweaks, replay-cache tuning, imputed-loss variants, polynomial learned-proposal hidden-size/epoch/warmup sweeps, or sparse pairwise count/gap sweeps as novelty.

Next Allowed:

- Local targets only with a materially different estimator or target construction, or with predeclared streaming/heldout generalization validation. Otherwise pivot compute to source-geometry objectives or less-prescriptive answer-derived boundary methods that reduce full forced-result enumeration.

Full Text:

```text
REVIEW: Local-target approximation is a ceiling, not the current scalable path.
Conclusion: Exact `policy_reweighted_t1` remains a useful proof of principle and diagnostic ceiling, but the tested scalable approximation families have now failed from enough angles to pause the branch as a mainline. Sparse uniform/top-k and adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, imputed sparse targets dilute pressure, simple learned proposals do not retain lift under streaming, random-prompt warmup is mixed-negative, and sparse pairwise preference failed even when `u32` covered the true result in `0.8450` of prompts (`0.0425` exact calc / `0.0234` sampled normal versus same-budget policy-reweighted `u32` at `0.3350` / `0.3438`).
Do not repeat: Do not run more sparse count ladders, top-k/neighborhood proposal tweaks, replay-cache tuning, imputed-loss variants, polynomial learned-proposal hidden-size/epoch/warmup sweeps, or sparse pairwise count/gap sweeps as novelty.
Next allowed test: Local targets only with a materially different estimator or target construction, or with predeclared streaming/heldout generalization validation. Otherwise pivot compute to source-geometry objectives or less-prescriptive answer-derived boundary methods that reduce full forced-result enumeration.
Source: `researchReviews/2026-05-30-local-target-approximation-direction-review.md`
```
