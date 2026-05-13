# 2026-05-13 - Next Task Planning After Result Feature Gate

Task:

```text
Analyze the current Phase 7 state and write the next best task document.
```

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `aiAgentWorkHistory/phase7/2026-05-13-result-space-interface-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-05-13-result-space-boundary-target-learning-signal.md`
- `aiAgentWorkHistory/phase7/2026-05-13-result-feature-separability-and-upstream-open-boundary-gate.md`
- Current implementation surface in `src/model.py` and
  `scripts/overfit_one_batch.py`

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-sixth-task-Full-grid-upstream-open-result-boundary-retention-gate.md
```

The selected direction is an exact full-grid upstream-open result-boundary
teaching and target-off retention gate. The task asks the next agent to add
`--exhaustive-grid-batch`, train on every ordered natural `0..19` pair exactly
once at every step, and then run retention only if the learned hard
calculator-result request crosses the Stage 1 gate.

## Rationale

The useful Phase 7 findings are:

- result-level action spaces removed same-sum pair ambiguity but did not make
  strict answer-loss Concrete discovery work;
- answer-derived forced-result targets are sharp and decoder/readout health is
  no longer the blocker;
- frozen exact result-head features contain nonlinear all-grid result
  information but are not linearly sufficient by threshold;
- frozen production heads failed, while upstream-open boundary teaching rose to
  `0.5975` hard result accuracy with semantic decoder movement exactly `0.0`.

The less useful next work would be oracle reruns, frozen-head schedule sweeps,
replication from failed checkpoints, canonical-query symmetry breaking, or
scaling beyond natural `0..19`.

The key implementation observation is that the current `batch_size=400`
training path samples random pairs each step; it is not guaranteed to be the
exact full `20 x 20` grid. Because the only near-positive branch drifted by
final, removing stochastic coverage noise is a better next test than jumping
immediately to a larger estimator implementation.

If exact-grid upstream-open teaching and the single allowed MLP rescue fail,
the task explicitly instructs Phase 7 to pivot to multi-sample result-space
policy gradient with per-prompt or leave-one-out baselines rather than more
boundary-target variants.

## Files Updated

- `aiAgentProjectTasks/2026-05-13-phase-7-sixth-task-Full-grid-upstream-open-result-boundary-retention-gate.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `SOLUTION_IDEAS.md`
- `aiAgentWorkHistory/phase7/2026-05-13-next-task-full-grid-upstream-open-boundary-retention-planning.md`
