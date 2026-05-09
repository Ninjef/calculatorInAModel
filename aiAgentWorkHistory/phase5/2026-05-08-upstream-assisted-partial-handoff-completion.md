# 2026-05-08 - Upstream-assisted partial-handoff completion

Task: Phase 5 second task, upstream-assisted completion from the known failed
seed `2`, Stage 1 step `55` handoff.

## Claim

Test whether upstream trainable parameters can help a below-boundary partially
taught calculator protocol recover after direct operand supervision is removed.
This is an upstream-assisted completion probe, not pure discovery from scratch.

## Runner

Added:

```text
scripts/run_phase5_upstream_assisted_partial_handoff_completion.py
```

Run root:

```text
runs/2026-05-08_phase5_upstream_assisted_partial_handoff_completion
```

The runner:

- records the source step `55` handoff and existing frozen baseline;
- runs the upstream-open answer-only continuation;
- writes `summary.json` and `summary.md`;
- compares final checkpoints against the source step `55` handoff by parameter
  group;
- runs canonical, private-protocol, and full-enum diagnostics on the final
  checkpoint and the selected post-recovery drift checkpoint.

## Starting point

- Source Stage 1 checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/checkpoint_snapshots/step_00055_weights.pt`
- Effective seed `2`, CLI seed `0`
- Source Stage 1 handoff operand exact `0.4375`
- Existing frozen-upstream failed continuation:
  `runs/2026-05-08_phase4_boundary_closure/stage2/seed2/step55/2026-05-08_072232_382505_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2`
- Retained upper-neighbor reference:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2`

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

| Condition | Final eval | Step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen step `55` baseline | `0.847656` | `0.84375` / `0.84375` / `0.84375` / `0.84375` | `0.000` | `0.015625` | `1.000` |
| retained step `60` reference | `1.000` | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |
| upstream-open step `55` | `0.998047` | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |

Selected upstream-open final checkpoint:

```text
runs/2026-05-08_phase5_upstream_assisted_partial_handoff_completion/stage1/upstream_open_lr3e-05/2026-05-08_170752_256015_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

## Parameter deltas

Compared with the source Stage 1 step `55` handoff:

| Group | L2 | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.input_proj` | `1.5496` | `0.1702` | `2/2` |
| `upstream_encoder` | `0.2829` | `0.01586` | `14/29` |
| `semantic_decoder` | `0.0` | `0.0` | `0/3` |

The upstream-open run was not a no-op: upstream parameters changed measurably
while semantic decoder parameters stayed unchanged.

## Diagnostics

Final diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| frozen step `55` baseline | `0.8555` / `0.8555` / `0.8555` | `0.8450` / `0.8450` / `0.8450` | `0.7051` / `0.7051` | `0.8516` |
| upstream-open final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |

Transient drift diagnostics:

| Selection | Fast gate | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream-open step `350` | `0.9609` | `0.9688` / `0.9688` / `0.9688` | `0.9500` / `0.9500` / `0.9500` | `0.3348` / `0.3348` | `0.9531` |

## Interpretation

This is a strong upstream-assisted completion positive from the known failed
seed `2`, Stage 1 step `55` handoff. The frozen-upstream baseline stayed
partial, while the upstream-open continuation recovered to retained-protocol
quality by the final checkpoint. Counterfactuals still show calculator
dependence, and direct teacher weights stayed exactly `0.0`.

This should not be described as pure upstream discovery. The run starts from a
partially taught protocol. It does show that opening upstream parameters can
cross a Phase 4 frozen-interface completion boundary while moving upstream
weights measurably.

The positive is final-checkpoint positive, not all-snapshot stable. The step
`350` drift checkpoint had positive full-enum gaps and degraded private/canonical
protocol metrics before the run recovered.

No optional lower-LR or anchor repeat was run because the primary run completed
and was diagnostically informative.

## Recommendation

Replicate upstream-assisted completion on another known failed handoff, likely
seed `5`, Stage 1 step `25` (`0.078125` operand exact). If that fails or shows
persistent drift, then test a minimal anchor or narrower unfreeze control.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_upstream_assisted_partial_handoff_completion.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py run --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py summarize
```
