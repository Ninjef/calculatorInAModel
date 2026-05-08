# Phase 5 Experiment Fact Sheet

## Direction

Phase 5 tests whether the Phase 4 taught-and-retained true calculator-query
protocol can survive, transfer, or improve once upstream/model-side parameters
are allowed to move. Stability and transfer-readiness claims are not pure
upstream-discovery claims.

## 2026-05-08 Upstream-Unfreeze Stability Smoke

Claim tested:

```text
Does allowing upstream parameters to move preserve a known retained
true-operand calculator protocol, or does it cause protocol drift?
```

Starting point:

- Retained Phase 4 seed `2`, Stage 2 step `60` checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- Source Stage 1 handoff:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/checkpoint_snapshots/step_00060_weights.pt`
- Effective seed `2`, CLI seed `0`
- Source Stage 1 handoff operand exact `0.640625`

Runner:

```text
scripts/run_phase5_upstream_unfreeze_stability_smoke.py
```

Run root:

```text
runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke
```

Shared setup:

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
- conservative `upstream_lr=0.00003`

Required continuations:

| Condition | Upstream frozen | Trainable groups |
| --- | ---: | --- |
| frozen control | yes | `calculator_hook.input_proj` (`1320`) |
| upstream open | no | `calculator_hook.input_proj` (`1320`), `upstream` (`4048`) |

Final fast gates:

| Condition | Final eval | Dense step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen control | `0.998047` (`511/512`) | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |
| upstream open | `0.998047` (`511/512`) | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |

Parameter deltas versus retained Phase 4 checkpoint:

| Condition | `calculator_hook.input_proj` L2 / max | upstream L2 / max | upstream tensors changed |
| --- | ---: | ---: | ---: |
| frozen control | `1.1914` / `0.1620` | `0.0` / `0.0` | `0/29` |
| upstream open | `1.6347` / `0.1554` | `0.2859` / `0.01814` | `14/29` |

Final diagnostics:

| Condition | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| frozen control final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |
| upstream-open final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |

Worst transient dense snapshots:

| Condition | Step | Fast gate normal/operand/pair/calc | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen control | `400` | `0.9258` / `0.9258` / `0.9258` / `0.9258` | `0.9648` / `0.9648` / `0.9648` | `0.9475` / `0.9475` / `0.9475` | `0.2793` / `0.2793` | `0.9531` |
| upstream open | `500` | `0.9414` / `0.9414` / `0.9414` / `0.9414` | `0.9414` / `0.9414` / `0.9414` | `0.9475` / `0.9475` / `0.9475` | `0.4017` / `0.4017` | `0.9375` |

Interpretation:

- The upstream-open final checkpoint is a cautious stability positive: final
  dense gates are exact, private diagnostics remain effectively exact, and
  full-enum learned-minus-true/best gaps are `0.0` while upstream parameters
  moved measurably.
- This is not a perfect all-snapshot stability result. Both the frozen control
  and upstream-open run had transient degraded dense snapshots; diagnostics on
  those checkpoints show real temporary protocol degradation and positive
  full-enum gaps.
- The upstream-open transient was slightly worse than the frozen transient by
  full-enum gap (`0.4017` vs `0.2793`), but it recovered by the final checkpoint.
- Do not label this as pure discovery. It supports using upstream-open
  continuations carefully, with dense checkpoint selection and diagnostics.

Recommendation:

Proceed to upstream-assisted partial-handoff completion from the known failed
seed `2`, Stage 1 step `55` handoff (`0.438` operand exact), but keep dense
snapshots and consider checkpoint selection rather than relying only on the
final checkpoint. A narrow-unfreeze or anchor control can be added if the
failed-handoff run shows persistent drift rather than transient dips.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_upstream_unfreeze_stability_smoke.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py run --jobs 2
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py diagnostics
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_unfreeze_stability_smoke.py summarize
```
