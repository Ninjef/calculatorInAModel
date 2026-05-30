# 2026-05-30 Result-Boundary Regret-Set Training Gate

## Question

Can a less-prescriptive answer-derived result-boundary target train the
calculator policy by allowing any near-best forced result, instead of forcing a
single argmin or spreading probability across all results with a softmax?

This tests a structural target-construction change after static soft targets
failed. It is still full-enum scoring, so it is a source-learning gate and not
yet a scalable recipe.

## Implementation

Added `--result-boundary-target-mode regret_set` to
`scripts/overfit_one_batch.py`.

For each prompt, the script scores all forced result classes, finds the best
answer NLL, and builds a uniform target over result classes with
`loss <= best_loss + margin`. The existing
`--result-boundary-target-temperature` flag supplies the margin for this mode.

Regression coverage:

```text
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "result_boundary_target_uses_lowest_nll_result or result_boundary_regret_set_targets_near_best_results or result_boundary_cli_validation"
```

Result:

```text
3 passed, 137 deselected
```

## Setup

Matched the known full-grid upstream-open result-boundary source gate:

- full `0..19` ordered-pair grid (`400` prompts)
- frozen product answer decoder
- upstream open
- `result_space` calculator head
- `operand_spans` read position
- `answer_loss_weight=0`
- `result_boundary_target_loss_weight=1`
- `steps=200`
- `lr=0.01`
- `upstream_lr=0.0003`
- snapshots/checkpoints every `50`

Probe runs:

```text
runs/2026-05-30_phase7_result_boundary_regret_set_probe/m005
runs/2026-05-30_phase7_result_boundary_regret_set_probe/m025
runs/2026-05-30_phase7_result_boundary_regret_set_probe/m1
runs/2026-05-30_phase7_result_boundary_regret_set_probe/m2
runs/2026-05-30_phase7_result_boundary_regret_set_probe/m4
```

Training gates:

```text
runs/2026-05-30_phase7_result_boundary_regret_set_training/hard_matched_step200
runs/2026-05-30_phase7_result_boundary_regret_set_training/m4_step200
```

## Target Width Probe

| Regret margin | Regret-set fraction | True in set | Effective results | True-result target mass |
| ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.0256` | `1.0000` | `1.0000` | `1.0000` |
| `0.25` | `0.0256` | `1.0000` | `1.0000` | `1.0000` |
| `1.0` | `0.0256` | `1.0000` | `1.0000` | `1.0000` |
| `2.0` | `0.0272` | `1.0000` | `1.0600` | `0.9700` |
| `4.0` | `0.1461` | `1.0000` | `5.6975` | `0.2413` |

Margins through `1.0` are effectively hard-best. Margin `4.0` is the first
meaningfully set-valued target, so it was the useful training gate.

## Training Results

Matched 200-step training:

| Target | Step-50 calc | Step-100 calc | Step-150 calc | Step-200 calc | Step-200 normal | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard-best `t=0.25` | `0.0975` | `0.1650` | `0.2475` | `0.4625` | `0.4600` | `0.4225` |
| regret-set margin `4.0` | `0.0425` | `0.0800` | `0.0825` | `0.0900` | `0.0875` | `0.0900` |

The exact hard comparator in this run is lower than an earlier matched hard
source (`0.5450` step-200 calc / `0.5475` final), but it still beats the
regret-set target by a wide margin under the same command path.

## Interpretation

The simple static regret-set target fails the source-acquisition gate.

- Narrow margins collapse to hard-best and are not a different method.
- The first nontrivial set target, margin `4.0`, contains about `5.7` allowed
  result classes and always contains the true result, but it dilutes the
  learning signal severely.
- This is a different failure mode from `soft_result`, but the practical
  conclusion is similar: static full-enum broad targets trade away too much
  pressure toward the useful result policy.

## Decision

```text
result_boundary_regret_set_training_negative
```

Do not continue static full-enum regret-set margin ladders on this source gate
as novelty. If set targets remain interesting, they need an adaptive/evolving
mechanism, calibrated proposal coupling, or a different credit-assignment
family rather than another static target over the same full-loss table.
