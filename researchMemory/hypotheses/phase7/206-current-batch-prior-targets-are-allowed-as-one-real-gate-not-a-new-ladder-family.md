# Current-batch prior targets are allowed as one real gate, not a new ladder family.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-06-03-current-batch-prior-target-review.md

Summary:

- Reviewed the just-added current-batch prior-target tooling against the closed route-excluded branch. It is distinct from route replay because it acts on the live batch, distinct from prior bootstrap because it writes no prompt-memory entries, and distinct from candidate-evidence/background refresh because it adds no new candidate scoring. That makes one real source gate strategically justified. But it still depends on a prior trained from prompt-memory targets, so it is not yet a non-prescriptive credit mechanism or a scalable many-calculator solution.

Questions:

- What did we learn about Current-batch prior targets are allowed as one real gate, not a new ladder family?
- Has Current-batch prior targets are allowed as one real gate, not a new ladder family been tested?
- Should we repeat Current-batch prior targets are allowed as one real gate, not a new ladder family?
- What is the status of Current-batch prior targets are allowed as one real gate, not a new ladder family?

Representative evidence:

- `researchReviews/2026-06-03-current-batch-prior-target-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run current-batch weight, confidence, route, cadence, or seed ladders as novelty if the first real source gate misses; do not reopen route replay, bootstrap, candidate-evidence, background-refresh, cap/fraction/window, or route-heldout diagnostic ladders under current-batch naming.

Next Allowed:

- Run one predeclared source gate for direct current-batch shared-prior target supply; if it misses heldout/excluded-route quality, move to joint/global target learning or less-prescriptive credit that bypasses answer-derived candidate scoring.

Full Text:

```text
REVIEW: Current-batch prior targets are allowed as one real gate, not a new ladder family.
Conclusion: Reviewed the just-added current-batch prior-target tooling against the closed route-excluded branch. It is distinct from route replay because it acts on the live batch, distinct from prior bootstrap because it writes no prompt-memory entries, and distinct from candidate-evidence/background refresh because it adds no new candidate scoring. That makes one real source gate strategically justified. But it still depends on a prior trained from prompt-memory targets, so it is not yet a non-prescriptive credit mechanism or a scalable many-calculator solution.
Do not repeat: Do not run current-batch weight, confidence, route, cadence, or seed ladders as novelty if the first real source gate misses; do not reopen route replay, bootstrap, candidate-evidence, background-refresh, cap/fraction/window, or route-heldout diagnostic ladders under current-batch naming.
Next allowed test: Run one predeclared source gate for direct current-batch shared-prior target supply; if it misses heldout/excluded-route quality, move to joint/global target learning or less-prescriptive credit that bypasses answer-derived candidate scoring.
Source: `researchReviews/2026-06-03-current-batch-prior-target-review.md`
```
