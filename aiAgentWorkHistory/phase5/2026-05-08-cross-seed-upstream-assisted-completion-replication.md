# 2026-05-08 - Cross-seed upstream-assisted completion replication

Task: Phase 5 third task, replicate upstream-assisted completion on the known
failed seed `5`, Stage 1 step `25` handoff.

## Claim

Test whether upstream-open answer-only continuation can complete the known
failed seed `5`, Stage 1 step `25` handoff after direct operand supervision is
removed. This is upstream-assisted partial-handoff completion, not pure
from-scratch discovery.

## Runner

Added:

```text
scripts/run_phase5_cross_seed_upstream_assisted_completion.py
```

Run root:

```text
runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion
```

The runner:

- records the source seed `5`, step `25` handoff and existing frozen baseline;
- records the retained seed `5`, step `30` upper-neighbor reference;
- runs the primary upstream-open answer-only continuation;
- supports the task's one optional lower-LR or anchor repeat;
- writes `summary.json` and `summary.md`;
- compares final checkpoints against the source handoff by parameter group;
- runs canonical, private-protocol, and full-enum diagnostics on final, best,
  and drift selections.

I also made the runner keep Torch imports out of the parent process before
training subprocesses start and set `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`
inside subprocess environments. This avoided an OpenMP shared-memory abort that
occurred before `overfit_one_batch.py` wrote training output.

## Starting point

- Source Stage 1 checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed5/2026-05-07_103539_395143_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5/checkpoint_snapshots/step_00025_weights.pt`
- Effective seed `5`, CLI seed `3`
- Source Stage 1 handoff operand exact `0.078125`
- Existing frozen-upstream failed continuation:
  `runs/2026-05-08_phase4_boundary_closure/stage2/seed5/step25/2026-05-08_072232_382502_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5`
- Retained upper-neighbor reference:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2_lower/seed5/step30/2026-05-07_173039_095164_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5`

## Shared setup

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=adaptive_interface`
- `freeze_semantic_decoder=true`
- `answer_loss_weight=1.0`
- `aux_operand_loss_weight=0.0`
- `adaptive_interface_loss_weight=0.0`
- primary `input_proj_anchor_weight=0.0`
- no oracle training
- `input_proj_lr=0.0003`
- `upstream_lr=0.00003`

## Fast gates

| Condition | Final eval | Step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen step `25` baseline | `0.865234` | `0.855469` / `0.855469` / `0.855469` / `0.855469` | `0.003906` | `0.019531` | `1.000` |
| retained step `30` reference | `0.994141` | `0.980469` / `0.980469` / `0.980469` / `0.980469` | `0.003906` | `0.019531` | `1.000` |
| upstream-open primary | `0.994141` | `0.996094` / `0.996094` / `0.996094` / `0.996094` | `0.003906` | `0.027344` | `1.000` |
| upstream-open anchor `0.001` | `0.972656` | `0.964844` / `0.964844` / `0.964844` / `0.964844` | `0.003906` | `0.027344` | `1.000` |

Selected checkpoints:

- Primary final:
  `runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/stage1/upstream_open_lr3e-05/2026-05-08_183823_224122_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed5/final_weights.pt`
- Primary best:
  `runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/stage1/upstream_open_lr3e-05/2026-05-08_183823_224122_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed5/checkpoint_snapshots/step_00950_weights.pt`
- Anchor final:
  `runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/stage2_optional_anchor/upstream_open_lr3e-05_anchor1e-03/2026-05-08_184246_899200_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-inanchor0.001-answer_decoder-sum_left_operand/model-c-2digit-seed5/final_weights.pt`
- Anchor best:
  `runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/stage2_optional_anchor/upstream_open_lr3e-05_anchor1e-03/2026-05-08_184246_899200_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-inanchor0.001-answer_decoder-sum_left_operand/model-c-2digit-seed5/checkpoint_snapshots/step_00800_weights.pt`

## Parameter deltas

Compared with the source Stage 1 step `25` handoff:

| Condition | `calculator_hook.input_proj` L2 / max | upstream L2 / max | upstream tensors changed | semantic decoder L2 |
| --- | ---: | ---: | ---: | ---: |
| upstream-open primary | `2.2578` / `0.1900` | `0.1909` / `0.01584` | `14/29` | `0.0` |
| upstream-open anchor `0.001` | `2.3284` / `0.1959` | `0.1964` / `0.01591` | `14/29` | `0.0` |

Both upstream-open runs were not no-ops: upstream parameters moved measurably
while the frozen semantic decoder stayed unchanged.

## Diagnostics

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| frozen step `25` baseline | `0.8281` / `0.8281` / `0.8281` | `0.8475` / `0.8475` / `0.8500` | `0.3492` / `0.3492` | `0.8828` |
| primary final step `1000` | `0.9922` / `0.9922` / `0.9922` | `0.9925` / `0.9925` / `0.9925` | `0.0601` / `0.0601` | `0.9844` |
| primary best step `950` | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` | `0.0` / `0.0` | `1.0000` |
| anchor final step `1000` | `0.9883` / `0.9883` / `0.9883` | `0.9775` / `0.9775` / `0.9775` | `0.2516` / `0.2516` | `0.9531` |
| anchor best step `800` | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` | `0.0` / `0.0` | `1.0000` |
| anchor drift step `900` | `0.9805` / `0.9805` / `0.9805` | `0.9475` / `0.9475` / `0.9475` | `0.4832` / `0.4832` | `0.9141` |

## Interpretation

This is a cross-seed upstream-assisted completion positive. The matched frozen
baseline stayed partial, while the primary upstream-open run reached retained
true-protocol quality at step `950`: canonical/private operand, pair, and
calculator-result accuracy were all `1.0`, and full-enum learned-minus-true and
learned-minus-best gaps were `0.0`.

The positive is checkpoint-selected, not all-snapshot stable. The primary final
checkpoint had mild drift (`0.0601` full-enum gaps), and the optional anchor
repeat reached an exact step `800` checkpoint but drifted more strongly by step
`900` and final. The anchor was therefore not a stabilization fix at weight
`0.001`.

Do not describe this as pure upstream discovery. It starts from a partially
taught seed `5`, step `25` protocol and removes direct teacher losses only for
the continuation.

## Recommendation

Phase 5 now has cross-seed evidence that opening upstream can complete failed
Phase 4 partial handoffs. The next task can consider a tightly bounded
from-scratch upstream-open scout, but success must remain checkpoint-selected
and diagnostic-heavy unless stability improves. A narrower unfreeze or other
stability control is also justified before broadening claims.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_cross_seed_upstream_assisted_completion.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py run --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py run-anchor --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py summarize
```
