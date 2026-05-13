# 2026-05-13 - Result Feature Separability Task Planning

Task:

```text
Analyze the current Phase 7 state and write the next best task document.
```

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- Phase 7 work histories for:
  - joint-pair result-group Stage 1;
  - result-space interface diagnostic;
  - result-space boundary-target learning signal.
- Current implementation surface around `result_space`, `result_proj`, and
  existing diagnostic probes in `src/model.py`,
  `scripts/overfit_one_batch.py`, and `scripts/diagnose_calculator_protocol.py`.

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fifth-task-Frozen-feature-result-separability-and-minimal-upstream-open-boundary-gate.md
```

The selected direction is a frozen-feature result separability gate followed by
the smallest conditional rescue:

1. Test whether the exact frozen operand-span features consumed by
   `calculator_hook.result_proj` can recover the answer-derived result target
   with a controlled linear probe or shallow MLP probe.
2. If linear separability passes, debug why the in-model result-boundary target
   did not learn.
3. If shallow capacity passes but linear fails, try a minimal MLP result head
   under the same boundary-target objective.
4. If frozen probes fail, run the minimal upstream-open boundary-target branch
   while keeping the semantic decoder frozen, then test target-off retention
   only if Stage 1 learns a real hard result protocol.

## Rationale

The useful Phase 7 finding is now sharper than "result-level action spaces did
not work." The project already made the target almost as easy as possible:
direct answer-derived CE over `0..38` result classes with only `result_proj`
trainable. That still failed. The target was valid and sharp, so the remaining
highest-value uncertainty is whether the frozen operand-span representation
contains the required result information for the current head.

Less useful next work would be more oracle/readout checks, more frozen linear
result-head LR or temperature sweeps, seed replication from failed checkpoints,
canonical-query symmetry breaking before result learning works, or scaling
beyond natural `0..19`.

This task gets to the end goal faster because it prevents the next experiments
from guessing blindly:

- A linear probe positive means the failure is likely in the in-model training
  path or optimization setup.
- A shallow-only positive means the model may need modest head capacity, not a
  new estimator family.
- A probe negative means strict frozen-upstream Phase 7 is exhausted, and the
  next real attempt should shape upstream representations or move to a
  different signal family.

## Files Updated

- `aiAgentProjectTasks/2026-05-13-phase-7-fifth-task-Frozen-feature-result-separability-and-minimal-upstream-open-boundary-gate.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentWorkHistory/phase7/2026-05-13-result-feature-separability-task-planning.md`
