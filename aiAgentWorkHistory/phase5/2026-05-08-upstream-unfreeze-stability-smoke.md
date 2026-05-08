# 2026-05-08 - Upstream-unfreeze stability smoke

Task: Phase 5 first task, upstream-unfreeze stability smoke.

## Claim

Test whether allowing upstream parameters to move preserves the known retained
Phase 4 seed `2`, Stage 2 step `60` true-operand calculator protocol, or causes
protocol drift. This is a stability and transfer-readiness probe, not a pure
discovery claim.

## Runner

Added:

```text
scripts/run_phase5_upstream_unfreeze_stability_smoke.py
```

Run root:

```text
runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke
```

The runner:

- verifies the retained Phase 4 checkpoint and source Stage 1 handoff exist;
- runs the matched frozen-upstream continuation;
- runs the conservative upstream-open continuation;
- writes `summary.json` and `summary.md`;
- compares final checkpoints against the retained Phase 4 checkpoint by
  parameter group;
- runs canonical, private-protocol, and full-enum diagnostics on both finals
  and the worst non-final dense snapshot for each condition.

## Starting point

- Retained checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- Source Stage 1 handoff:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/checkpoint_snapshots/step_00060_weights.pt`
- Effective seed `2`, CLI seed `0`
- Stage 1 handoff operand exact `0.640625`

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
- `input_proj_anchor_weight=0.0`
- no oracle training
- `input_proj_lr=0.0003`
- `upstream_lr=0.00003`

## Fast gates

| Condition | Run path | Final eval | Step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen control | `runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke/stage1/frozen_upstream_control/2026-05-08_110738_867165_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `0.998047` | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |
| upstream open | `runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke/stage2/upstream_open_lr3e-05/2026-05-08_110738_867389_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `0.998047` | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |

Final checkpoints:

- frozen control:
  `runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke/stage1/frozen_upstream_control/2026-05-08_110738_867165_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- upstream open:
  `runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke/stage2/upstream_open_lr3e-05/2026-05-08_110738_867389_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`

Both runs had:

- `final_aux_operand_loss_weight=0.0`
- `final_adaptive_interface_loss_weight=0.0`
- `final_input_proj_anchor_weight=0.0`
- `freeze_semantic_decoder=true`

## Parameter deltas

Compared with the retained Phase 4 checkpoint:

| Condition | Trainable groups | Input-proj L2 / max | Upstream L2 / max | Upstream tensors changed |
| --- | --- | ---: | ---: | ---: |
| frozen control | `calculator_hook.input_proj` (`1320`) | `1.1914` / `0.1620` | `0.0` / `0.0` | `0/29` |
| upstream open | `calculator_hook.input_proj` (`1320`), `upstream` (`4048`) | `1.6347` / `0.1554` | `0.2859` / `0.01814` | `14/29` |

The upstream-open run was not a no-op: upstream parameters changed measurably
while semantic decoder parameters stayed unchanged.

## Diagnostics

Final diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| frozen final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |
| upstream-open final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |

Worst transient snapshots:

| Selection | Fast gate | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen step `400` | `0.9258` | `0.9648` / `0.9648` / `0.9648` | `0.9475` / `0.9475` / `0.9475` | `0.2793` / `0.2793` | `0.9531` |
| upstream-open step `500` | `0.9414` | `0.9414` / `0.9414` / `0.9414` | `0.9475` / `0.9475` / `0.9475` | `0.4017` / `0.4017` | `0.9375` |

## Interpretation

The upstream-open final checkpoint preserves the learned true-operand protocol
at near-Phase-4 retained quality while upstream parameters move measurably.
Full-enum gaps are `0.0`, and private all-pair decoding remains effectively
exact.

This is not a perfect all-snapshot stability result. Both conditions had
transient degraded dense snapshots, and the transient diagnostics show real
temporary protocol degradation rather than a logging artifact. The upstream
open transient was slightly worse by full-enum gap, but it recovered by the
final checkpoint.

## Recommendation

Proceed to the next Phase 5 task: upstream-assisted partial-handoff completion
from the known failed seed `2`, Stage 1 step `55` handoff (`0.438` operand
exact). Keep dense snapshots, select checkpoints by learned-interface metrics,
and consider a narrow-unfreeze or anchor control if the failed-handoff run
shows persistent drift.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_upstream_unfreeze_stability_smoke.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py run --jobs 2
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py diagnostics
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py summarize
```
