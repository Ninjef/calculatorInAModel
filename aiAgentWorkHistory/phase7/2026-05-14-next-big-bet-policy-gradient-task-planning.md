# 2026-05-14 - Next Big-Bet Policy-Gradient Task Planning

## Task

Analyze the current Phase 7 state and write the next best task document, with
attention to what has been helpful, what has become repetitive, and how to move
the project toward the core calculator-use goal quickly.

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `factSheets/PHASE_1_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentWorkHistory/phase7/2026-05-13-full-grid-upstream-open-result-boundary-retention-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-13-exact-grid-retained-positive-seed-replication.md`
- Current implementation surface in `src/model.py`, `scripts/overfit_one_batch.py`, and `tests/test_model.py`

## Synthesis

Helpful Phase 7 knowledge:

- Exact-grid coverage was decisive for the upstream-open boundary-target
  branch.
- The natural product decoder and result landscape are settled infrastructure.
- Boundary-target teaching can produce near-exact hard result requests across
  seeds with semantic decoder movement exactly `0.0`.
- Dense exact-grid snapshots, full-enum learned-result diagnostics, and
  injection/forced-random controls are the right selection discipline.

Less helpful as next work:

- More oracle/readout checks.
- More random-resampled or frozen-head boundary-target variants.
- More target-off retention reruns that do not introduce a new mechanism.
- Canonical-query/protocol stabilization before robust retention or a stronger
  discovery signal exists.

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
```

The selected direction is multi-sample result-space policy-gradient training
with per-prompt or leave-one-out baselines, plus a Stage 0 gradient-agreement
diagnostic against the known boundary-target ceiling.

## Rationale

The seed-replication result changed the plan. Exact-grid boundary-target
teaching is robust, but target-off retention is seed-fragile under the strict
gate. That means the project should stop iterating on small boundary-target
capacity/schedule variants and instead test a genuinely different
non-differentiable learning signal.

Policy gradient is the most direct next big bet because it trains from the
real sampled calculator action's answer loss. To avoid repeating Phase 1, the
new task explicitly requires result-space actions, exact-grid batches,
multi-sample per-prompt baselines, and gradient-alignment checks before
long-run training.

## Files Updated

- `aiAgentProjectTasks/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `SOLUTION_IDEAS.md`
