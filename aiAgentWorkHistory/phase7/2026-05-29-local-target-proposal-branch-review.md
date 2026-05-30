# 2026-05-29 Local-Target Proposal Branch Review

## Goal

Perform the requested periodic zoom-out after the local-target branch accrued
multiple approximation attempts, and decide whether future agents should keep
running nearby proposal variants.

## Inputs Reviewed

- `RESEARCH_STATE.md`
- `HYPOTHESIS_LEDGER.md`
- `researchMemory/phase7-direction-memory.md`
- `researchReviews/2026-05-29-phase7-local-target-approximation-review.md`
- `researchReviews/2026-05-29-replay-memory-branch-review.md`
- `aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md`

## Decision

```text
simple_local_target_proposal_approximation_paused
```

## Summary

Exact `policy_reweighted_t1` remains a useful ceiling and proves that
answer-derived local targets can train result-level calculator use.

Simple approximation mechanisms should stop as mainline work:

- sparse/top-k/adaptive candidates need near-full coverage;
- fixed replay memory is prompt-transductive and fails streaming stress;
- mean/current/max imputed unscored-mass correction dilutes pressure;
- the simple learned proposal wins fixed-grid but not streaming;
- random-prompt proposal pretraining does not create robust functional
  streaming lift.

Future local-target work needs a different estimator, different target
construction, or a learned proposal validated explicitly on
streaming/full-grid generalization. Otherwise, compute should move back to
source objectives that improve actual 600-step additive handoff/readout
behavior.
