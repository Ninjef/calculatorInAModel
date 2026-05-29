# Fixed replay-memory local-target proposals are not the scalable path.

Kind: hypothesis_memory
Status: PAUSED
Phase: Phase 7
Source: researchReviews/2026-05-29-replay-memory-branch-review.md

Summary:

- The replay-memory branch produced a real fixed-grid positive, but the follow-up stress tests identify the mechanism as prompt-identity transduction rather than a scalable candidate proposal. Lower fresh budgets worked on the fixed grid, but rescoring did not improve retention, reset windows damaged learning, and streaming minibatches removed the strong lift. This pauses fixed per-prompt replay caches as a family, including fresh-count, rescore-count, reset-interval, batch-size, and longer-run variants.

Questions:

- What did we learn about Fixed replay-memory local-target proposals are not the scalable path?
- Has Fixed replay-memory local-target proposals are not the scalable path been tested?
- Should we repeat Fixed replay-memory local-target proposals are not the scalable path?
- What is the status of Fixed replay-memory local-target proposals are not the scalable path?
- What follow-up is allowed for Fixed replay-memory local-target proposals are not the scalable path?

Representative evidence:

- `researchReviews/2026-05-29-replay-memory-branch-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more fixed replay-memory budget ladders, rescore ladders, reset intervals, streaming batch-size/length checks, or seed replications as novelty.

Next Allowed:

- Local-target work needs a genuinely new mechanism: learned/generalized candidate proposal, estimator/bias correction, or a target construction that does not require the useful result to already be in a hand-coded candidate set. Otherwise prioritize source objectives that improve additive handoff/readout geometry.

Full Text:

```text
PAUSED: Fixed replay-memory local-target proposals are not the scalable path.
Conclusion: The replay-memory branch produced a real fixed-grid positive, but the follow-up stress tests identify the mechanism as prompt-identity transduction rather than a scalable candidate proposal. Lower fresh budgets worked on the fixed grid, but rescoring did not improve retention, reset windows damaged learning, and streaming minibatches removed the strong lift. This pauses fixed per-prompt replay caches as a family, including fresh-count, rescore-count, reset-interval, batch-size, and longer-run variants.
Do not repeat: Do not run more fixed replay-memory budget ladders, rescore ladders, reset intervals, streaming batch-size/length checks, or seed replications as novelty.
Next allowed test: Local-target work needs a genuinely new mechanism: learned/generalized candidate proposal, estimator/bias correction, or a target construction that does not require the useful result to already be in a hand-coded candidate set. Otherwise prioritize source objectives that improve additive handoff/readout geometry.
Source: `researchReviews/2026-05-29-replay-memory-branch-review.md`
```
