# Simple local-target proposal approximation is not the scalable path.

Kind: hypothesis_memory
Status: PAUSED
Phase: Phase 7
Source: researchReviews/2026-05-29-local-target-proposal-branch-review.md

Summary:

- Reviewing the local-target approximation cluster shows a consistent failure mode. Exact `policy_reweighted_t1` remains a useful ceiling and proof of principle, and replay/learned proposal variants produced fixed-grid positives, but simple proposal mechanisms did not survive scalability stress: raw/top-k/adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, unscored-mass imputation diluted pressure, the online learned proposal tied raw `u32` under 800-step streaming, and random-prompt proposal pretraining hurt sampled normal despite a small exact-calc nudge.

Questions:

- What did we learn about Simple local-target proposal approximation is not the scalable path?
- Has Simple local-target proposal approximation is not the scalable path been tested?
- Should we repeat Simple local-target proposal approximation is not the scalable path?
- What is the status of Simple local-target proposal approximation is not the scalable path?
- What follow-up is allowed for Simple local-target proposal approximation is not the scalable path?

Representative evidence:

- `researchReviews/2026-05-29-local-target-proposal-branch-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more raw count ladders, top-k/neighborhood variants, fixed replay cache variants, mean/current/max imputation branches, or the same polynomial-feature learned proposal with or without `_wN` warmup as novelty.

Next Allowed:

- Local targets need a different estimator, a different target construction, or a learned proposal whose validation objective explicitly targets streaming/full-grid generalization. Otherwise prioritize source objectives aimed at actual 600-step additive handoff/readout behavior.

Full Text:

```text
PAUSED: Simple local-target proposal approximation is not the scalable path.
Conclusion: Reviewing the local-target approximation cluster shows a consistent failure mode. Exact `policy_reweighted_t1` remains a useful ceiling and proof of principle, and replay/learned proposal variants produced fixed-grid positives, but simple proposal mechanisms did not survive scalability stress: raw/top-k/adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, unscored-mass imputation diluted pressure, the online learned proposal tied raw `u32` under 800-step streaming, and random-prompt proposal pretraining hurt sampled normal despite a small exact-calc nudge.
Do not repeat: Do not run more raw count ladders, top-k/neighborhood variants, fixed replay cache variants, mean/current/max imputation branches, or the same polynomial-feature learned proposal with or without `_wN` warmup as novelty.
Next allowed test: Local targets need a different estimator, a different target construction, or a learned proposal whose validation objective explicitly targets streaming/full-grid generalization. Otherwise prioritize source objectives aimed at actual 600-step additive handoff/readout behavior.
Source: `researchReviews/2026-05-29-local-target-proposal-branch-review.md`
```
