# Local-target approximation is a ceiling, not the current scalable path

Status: REVIEW / PAUSED

Exact `policy_reweighted_t1` remains a valuable proof of principle: it trains a
natural result-level calculator policy and can survive answer-only retention.
But the approximation cluster has now failed from several angles. Sparse
uniform/top-k proposals need near-full result coverage, adaptive neighborhoods
underperform raw uniform `u32`, fixed replay memory is prompt-transductive,
unscored-mass imputation dilutes target pressure, simple learned proposals lose
their lift under streaming minibatches, random-prompt warmup is mixed-negative,
and sparse pairwise preferences fail even at high true-candidate coverage.

Key new evidence: `sampled_pairwise_preference_u32` saw the true result in
`0.8450` of prompts but reached only `0.0425` exact-grid calculator accuracy
and `0.0234` sampled normal, while same-budget
`sampled_policy_reweighted_t1_k0_u32` reached `0.3350` / `0.3438`.

Do not repeat: no more sparse count ladders, top-k/neighborhood proposal
tweaks, replay-cache tuning, imputed-loss variants, polynomial learned-proposal
hidden-size/epoch/warmup sweeps, or sparse pairwise count/gap sweeps as
novelty.

Next allowed test: local targets only with a materially different estimator or
target construction, or with predeclared streaming/heldout generalization
validation. Otherwise pivot compute to source-geometry objectives or
less-prescriptive answer-derived boundary methods that reduce full
forced-result enumeration.

Source:
`researchReviews/2026-05-30-local-target-approximation-direction-review.md`
