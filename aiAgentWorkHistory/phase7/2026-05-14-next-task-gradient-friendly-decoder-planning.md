# 2026-05-14 - Next Task Gradient-Friendly Decoder Planning

## Task

Analyze the current Phase 7 state and write the next best task document, with
attention to what has been helpful, what has become low-value repetition, and
how to move quickly toward the core calculator-use thesis.

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentWorkHistory/phase7/2026-05-13-next-learning-signal-task-planning.md`
- `aiAgentWorkHistory/phase7/2026-05-14-next-big-bet-policy-gradient-task-planning.md`
- `aiAgentWorkHistory/phase7/2026-05-14-multisample-result-space-policy-gradient-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`
- Completed Phase 7 result-space, boundary-target, policy-gradient, and exact
  result-marginal task documents.
- Current implementation surface in `src/model.py`,
  `scripts/overfit_one_batch.py`, and `tests/test_model.py`.

## Synthesis

Helpful Phase 7 knowledge:

- Exact-grid coverage is worth keeping. It turned a partial upstream-open
  boundary-target result into near-exact Stage 1 learning across seeds.
- The boundary-target branch is the best supervised ceiling/control for new
  mechanisms because it learns hard natural result requests with semantic
  decoder movement exactly `0.0`.
- Target-off retention diagnostics remain useful as stability probes, but not
  as novelty by themselves.
- The result-space REINFORCE and exact result-marginal implementations are
  useful diagnostic infrastructure even though their raw objective failed.
- The three-way gradient gate is now the right filter before any long run:
  candidate gradient, exact expected-cost/PG reference, and boundary-target
  ceiling on the same exact grid.

Low-value next work:

- More oracle/readout checks for the existing Phase 6 product decoder.
- Random-resampled or frozen-head boundary-target reruns.
- More target-off retention reruns that do not introduce a new mechanism or
  diagnose the known seed fragility.
- Canonical-query stabilization before a robust natural result request exists.
- Longer vanilla result-space PG, actor-critic, learned-baseline, or
  RELAX/NVIL runs that merely estimate the same raw expected-cost gradient.
- Raw exact result-marginal expected-cost training with the same frozen
  decoder.

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
```

The selected direction is a stricter downstream decoder/loss-geometry gate.
The task asks whether a result-calibrated frozen decoder can make exact
answer-loss gradients over result actions align with the boundary-target
ceiling before any long model-side training.

## Rationale

The exact result-marginal gate changed the research fork. It showed that the
sampled PG failure was not mainly finite-sample variance: sampled PG was
strongly aligned with the raw exact expected-cost gradient, and that exact
gradient was anti-aligned with the boundary-target ceiling.

That makes ordinary variance reduction a weak next bet. The current frozen
decoder can answer correctly when forced the true result, but its local
answer-loss geometry is not a good teacher for the model-side result request.
The fastest honest test is therefore not another estimator for the same
objective; it is to ask whether the downstream decoder can be trained to be
gradient-friendly while remaining frozen during upstream discovery.

If the decoder alignment gate passes, Stage 1 can test exact result-marginal
answer-loss discovery with no true-result labels, no boundary-target updates,
no oracle operands, and no semantic decoder movement. If it fails, Phase 7
should pivot to explicitly biased backward channels such as synthetic
gradients, direct feedback alignment, or learned shadow-gradient modules.

## Files Updated

- `aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `SOLUTION_IDEAS.md`
- `CLAUDE.md`
