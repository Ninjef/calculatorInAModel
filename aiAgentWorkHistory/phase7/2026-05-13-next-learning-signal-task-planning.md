# 2026-05-13 - Next Learning-Signal Task Planning

Task:

```text
Analyze the current Phase 7 state and write the next best task document.
```

## Context Reviewed

- `CLAUDE.md`
- `OVERARCHING_EXPERIMENT_PURPOSE.md`
- `SOLUTION_IDEAS.md`
- `docs/canonical_diagnostics.md`
- `factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- Phase 7 overarching plan
- prior Phase 1 REINFORCE work history
- current implementation surface in `src/model.py` and
  `scripts/overfit_one_batch.py`

## Decision

Created the next task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

The selected direction is an answer-derived result boundary-target objective:
enumerate calculator result classes through the frozen product decoder, train
the `result_space` request head toward the best answer-NLL result, then turn
the boundary-target objective exactly off and test retention.

## Rationale

The useful Phase 7 finding is that result-level action parameterization alone
is not enough. Joint-pair and direct result-space deterministic Concrete both
failed from strict initialization, even though the frozen product decoder and
full-enum result landscape remain healthy.

The less useful next work would be more small Concrete schedule sweeps, oracle
reruns, pair-exact optimization for natural sum-only addition, or replication
from failed checkpoints.

The fastest high-signal next learning signal is target propagation / local
boundary targets because Phase 6 already showed that answer-derived local
targets can teach and hand off protocols in an identifiable setting. The
boundary-target task asks whether that kind of signal can create a natural
result-level protocol and whether answer-only continuation can keep it after
the target is removed.

If this task passes, Phase 7 gets a natural result-level calculator-use
positive with teacher removal. If it fails after Stage 1, the project should
pivot to multi-sample policy gradient with per-prompt baselines, surrogate
gradients, or direct feedback alignment rather than more deterministic Concrete
tweaks.

## Files Updated

- `aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`
- `aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md`
- `SOLUTION_IDEAS.md`
