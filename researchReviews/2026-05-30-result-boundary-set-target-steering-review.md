# 2026-05-30 Result-Boundary Set-Target Steering Review

## Why This Review Exists

The previous static result-boundary review left one plausible target-side
escape hatch: set targets tied to uncertainty or regret, rather than
temperature-softened probability over all result classes. The `regret_set`
gate tested the simplest version of that idea.

## What Changed

- `regret_set` target mode is implemented and unit tested.
- Narrow regret margins (`0.05`, `0.25`, `1.0`) collapsed to hard-best targets.
- Margin `2.0` was still almost hard-best (`1.06` effective results).
- Margin `4.0` was genuinely set-valued (`5.6975` effective results, true
  result target mass `0.2413`, true result always in set), but trained much
  worse than a matched hard-best comparator:
  - hard-best step-200 learned calc / final eval: `0.4625` / `0.4225`
  - regret-set margin `4.0` step-200 learned calc / final eval: `0.0900` /
    `0.0900`

## What Should Stop

Do not spend mainline compute on more static full-enum result-boundary target
ladders:

- `soft_result` temperature ladders;
- fixed regret-margin ladders;
- simple "top-N low regret" variants over the same static full-loss table;
- other broad static targets whose main effect is to reduce pressure toward
  the best/true result.

These variants are now a local rut, not a route toward the thesis.

## What Still Deserves Compute

Result-boundary remains useful as an answer-derived bridge and benchmark, but
future work needs a mechanism change:

- evolving-checkpoint or streaming validation where proposal quality is tested
  under changing model states;
- calibrated proposal learning that reduces scoring while preserving target
  quality;
- set targets only if coupled to adaptive uncertainty/regret selection rather
  than a fixed full-enum margin;
- a different credit-assignment family that can pass a local feasibility gate
  and then show early Stage 1 lift.

## Decision

```text
static_result_boundary_set_targets_paused
```

The project is not closer to the full goal by further tuning static set targets.
The next work should either make result-boundary adaptive/evolving, or move to
a different less-prescriptive credit-assignment mechanism.
