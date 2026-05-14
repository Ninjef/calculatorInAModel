# 2026-05-14 - Next Exact Result-Marginal Task Planning

## Task

Analyze the current Phase 7 state and write the next best task document after
the multi-sample result-space policy-gradient gate.

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentWorkHistory/phase7/2026-05-14-multisample-result-space-policy-gradient-gate.md`
- `aiAgentProjectTasks/completed/phase7/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md`
- Relevant implementation surfaces in `src/model.py`,
  `scripts/overfit_one_batch.py`, and `tests/test_model.py`

## Synthesis

Helpful Phase 7 knowledge:

- Exact-grid coverage is an important stabilizer and should remain the default
  for natural `0..19` result-level gates.
- The product answer decoder/readout is healthy infrastructure, not a current
  research bottleneck.
- Upstream-open boundary-target teaching proves that natural result requests
  are representable and teachable with semantic decoder movement exactly `0.0`.
- The boundary-target branch is now most useful as a supervised ceiling/control
  for comparing new learning signals.
- Result-space REINFORCE plumbing is useful even though the vanilla estimator
  failed: it gives a sampled score-function gradient to compare against exact
  result-marginal gradients.

Less helpful as next work:

- Oracle/readout reruns.
- Random-resampled or frozen-head boundary-target variants.
- More target-off retention reruns without a new mechanism.
- Canonical-query stabilization before robust retention or stronger discovery.
- Vanilla multi-sample result-space PG long runs while the Stage 0 gradient
  cosine remains negative.
- Jumping straight to learned-baseline score-function methods before checking
  whether the exact expected result-action gradient is aligned.

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
```

The selected direction is an exact result-marginal answer-loss gradient gate:
enumerate the `0..38` natural result actions, compute answer NLL for each
forced result, form the exact expected answer-loss gradient under the model's
result policy, and compare that gradient against both sampled REINFORCE and
the boundary-target ceiling on the exact `20 x 20` grid.

## Rationale

The policy-gradient negative is not yet enough to pick the next estimator
family. A learned baseline can reduce variance, but it cannot fix a
fundamentally misaligned expected-cost objective. The exact result-marginal
gate is the shortest diagnostic that separates those two cases.

If exact result-marginal gradients align but sampled PG does not, the project
should treat vanilla PG as a variance/control-variate negative and can train
with exact enumeration while the result action space is small. If exact
result-marginal gradients are also negative or near-zero, the project should
stop spending effort on expected-cost/score-function variants and pivot to
surrogate gradients, synthetic gradients/direct feedback alignment, or a
stricter decoder-phase bottleneck.

## Files Updated

- `aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
