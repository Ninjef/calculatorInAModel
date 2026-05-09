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

## 2026-05-08 Upstream-Assisted Partial-Handoff Completion

Claim tested:

```text
Can upstream trainable parameters help a below-boundary partially taught
calculator protocol recover after direct operand supervision is removed?
```

Starting point:

- Source Stage 1 seed `2`, step `55` checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/checkpoint_snapshots/step_00055_weights.pt`
- Effective seed `2`, CLI seed `0`
- Source Stage 1 operand exact `0.4375`
- Existing frozen-upstream failed continuation:
  `runs/2026-05-08_phase4_boundary_closure/stage2/seed2/step55/2026-05-08_072232_382505_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2`
- Retained upper-neighbor reference:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2`

Runner:

```text
scripts/run_phase5_upstream_assisted_partial_handoff_completion.py
```

Run root:

```text
runs/2026-05-08_phase5_upstream_assisted_partial_handoff_completion
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
- `upstream_lr=0.00003`

Primary continuation:

| Condition | Upstream frozen | Trainable groups |
| --- | ---: | --- |
| upstream open from step `55` | no | `calculator_hook.input_proj` (`1320`), `upstream` (`4048`) |

Fast gates:

| Condition | Final eval | Dense step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing frozen step `55` failure | `0.847656` | `0.84375` / `0.84375` / `0.84375` / `0.84375` | `0.000` | `0.015625` | `1.000` |
| retained step `60` reference | `1.000` | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |
| upstream-open step `55` | `0.998047` (`511/512`) | `1.000` / `1.000` / `1.000` / `1.000` | `0.000` | `0.015625` | `1.000` |

Parameter deltas versus the source step `55` handoff:

| Condition | `calculator_hook.input_proj` L2 / max | upstream L2 / max | upstream tensors changed | semantic decoder L2 |
| --- | ---: | ---: | ---: | ---: |
| upstream-open step `55` | `1.5496` / `0.1702` | `0.2829` / `0.01586` | `14/29` | `0.0` |

Final diagnostics:

| Condition | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| existing frozen step `55` failure | `0.8555` / `0.8555` / `0.8555` | `0.8450` / `0.8450` / `0.8450` | `0.7051` / `0.7051` | `0.8516` |
| upstream-open step `55` final | `0.9961` / `0.9961` / `0.9961` | `0.9975` / `0.9975` / `0.9975` | `0.0` / `0.0` | `1.000` |

Transient drift diagnostic:

| Selection | Fast gate normal/operand/pair/calc | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream-open step `350` | `0.9609` / `0.9609` / `0.9609` / `0.9609` | `0.9688` / `0.9688` / `0.9688` | `0.9500` / `0.9500` / `0.9500` | `0.3348` / `0.3348` | `0.9531` |

Interpretation:

- This is a strong upstream-assisted completion positive from the known failed
  seed `2`, step `55` handoff. The matched frozen-upstream continuation stayed
  partial, while the upstream-open continuation recovered to retained-protocol
  quality by the final checkpoint.
- The result is not pure discovery from scratch: it starts from a partially
  taught Stage 1 protocol and removes the direct teacher only for the
  continuation.
- The result is not a no-op: upstream parameters moved measurably while the
  semantic decoder stayed unchanged.
- The run still had transient protocol degradation after reaching exact fast
  gates. Step `350` had positive full-enum gaps and private/canonical protocol
  degradation before recovering by the final checkpoint.
- No optional lower-LR or anchor repeat was run because the primary run was
  informative and reached final retained-protocol quality.

Recommendation:

Replicate upstream-assisted completion on another known failed handoff, likely
seed `5`, Stage 1 step `25` (`0.078125` operand exact), before broadening into
new unfreeze controls or new estimators.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_upstream_assisted_partial_handoff_completion.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py run --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase5_upstream_assisted_partial_handoff_completion.py summarize
```

## 2026-05-08 Cross-Seed Upstream-Assisted Completion Replication

Claim tested:

```text
Can upstream-open answer-only continuation complete the known failed seed 5,
Stage 1 step 25 handoff after direct operand supervision is removed?
```

Starting point:

- Source Stage 1 seed `5`, step `25` checkpoint:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed5/2026-05-07_103539_395143_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5/checkpoint_snapshots/step_00025_weights.pt`
- Effective seed `5`, CLI seed `3`
- Source Stage 1 operand exact `0.078125`
- Existing frozen-upstream failed continuation:
  `runs/2026-05-08_phase4_boundary_closure/stage2/seed5/step25/2026-05-08_072232_382502_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5`
- Retained upper-neighbor reference:
  `runs/2026-05-07_phase4_min_supervision_boundary/stage2_lower/seed5/step30/2026-05-07_173039_095164_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5`

Runner:

```text
scripts/run_phase5_cross_seed_upstream_assisted_completion.py
```

Run root:

```text
runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion
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
- primary `input_proj_anchor_weight=0.0`
- no oracle training
- `input_proj_lr=0.0003`
- `upstream_lr=0.00003`

Fast gates:

| Condition | Final eval | Dense step 1000 normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing frozen step `25` failure | `0.865234` | `0.855469` / `0.855469` / `0.855469` / `0.855469` | `0.003906` | `0.019531` | `1.000` |
| retained step `30` reference | `0.994141` | `0.980469` / `0.980469` / `0.980469` / `0.980469` | `0.003906` | `0.019531` | `1.000` |
| upstream-open step `25` primary | `0.994141` | `0.996094` / `0.996094` / `0.996094` / `0.996094` | `0.003906` | `0.027344` | `1.000` |
| upstream-open step `25` anchor `0.001` | `0.972656` | `0.964844` / `0.964844` / `0.964844` / `0.964844` | `0.003906` | `0.027344` | `1.000` |

Parameter deltas versus the source step `25` handoff:

| Condition | `calculator_hook.input_proj` L2 / max | upstream L2 / max | upstream tensors changed | semantic decoder L2 |
| --- | ---: | ---: | ---: | ---: |
| upstream-open primary | `2.2578` / `0.1900` | `0.1909` / `0.01584` | `14/29` | `0.0` |
| upstream-open anchor `0.001` | `2.3284` / `0.1959` | `0.1964` / `0.01591` | `14/29` | `0.0` |

Diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| frozen step `25` baseline | `0.8281` / `0.8281` / `0.8281` | `0.8475` / `0.8475` / `0.8500` | `0.3492` / `0.3492` | `0.8828` |
| primary final step `1000` | `0.9922` / `0.9922` / `0.9922` | `0.9925` / `0.9925` / `0.9925` | `0.0601` / `0.0601` | `0.9844` |
| primary best step `950` | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` | `0.0` / `0.0` | `1.0000` |
| anchor final step `1000` | `0.9883` / `0.9883` / `0.9883` | `0.9775` / `0.9775` / `0.9775` | `0.2516` / `0.2516` | `0.9531` |
| anchor best step `800` | `1.0000` / `1.0000` / `1.0000` | `1.0000` / `1.0000` / `1.0000` | `0.0` / `0.0` | `1.0000` |
| anchor drift step `900` | `0.9805` / `0.9805` / `0.9805` | `0.9475` / `0.9475` / `0.9475` | `0.4832` / `0.4832` | `0.9141` |

Interpretation:

- This replicates upstream-assisted partial-handoff completion on a second
  seed and a much lower Stage 1 handoff. The primary upstream-open run crossed
  from the frozen baseline's partial protocol to an exact selected checkpoint
  at step `950`.
- The result is checkpoint-selected completion, not all-snapshot stability. The
  primary final checkpoint had mild protocol drift, and the optional anchor
  repeat reached an exact step `800` checkpoint but drifted more strongly by
  step `900` and final.
- The result is not pure discovery from scratch. It starts from a partially
  taught seed `5`, step `25` protocol and removes direct teacher losses only in
  the continuation.
- The result is not a no-op: upstream tensors moved measurably in both runs,
  while semantic decoder tensors stayed unchanged.

Recommendation:

Phase 5 now has cross-seed evidence that opening upstream can complete failed
Phase 4 partial handoffs, but stability still depends on checkpoint selection.
The next task can either run a tightly bounded from-scratch upstream-open scout
or first test a narrower stability control; in either case, keep dense
snapshots and canonical/private/full-enum diagnostics as the success criterion.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase5_cross_seed_upstream_assisted_completion.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py run --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py run-anchor --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_cross_seed_upstream_assisted_completion.py summarize
```

## 2026-05-09 No-Handoff Upstream Discovery Smoke

Claim tested:

```text
Can answer-only training discover the calculator-query protocol without any
Stage 1 supervised interface handoff, while preserving the strict Phase 4/5
semantic bottleneck?
```

Implementation facts:

- Added `--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only`
  to `scripts/overfit_one_batch.py`.
- The default remains `full_model`, preserving historical behavior.
- The new `semantic_decoder_only` opt-in loads only `answer_offset_emb.*`,
  `answer_decoder.*`, and `calculator_hook.output_proj.*`.
- This smoke explicitly used `semantic_decoder_checkpoint_load_scope=full_model`.
- Current loader behavior is recorded as
  `semantic_decoder_checkpoint_load_scope_current_behavior=full_model_current_behavior`.

Runner:

```text
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
```

Run root:

```text
runs/2026-05-09_phase5_no_handoff_upstream_discovery_smoke
```

Starting point:

- Stage 0B operand-aware oracle semantic decoder checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- The corresponding repo-local checkpoint was not present, so the absolute
  Phase 4 fact-sheet path was used.
- Interpretation label: `no_handoff_full_model_init`.
- This is not strict random-upstream discovery because full-model checkpoint
  loading starts from the Stage 0B upstream state.

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
- `input_proj_lr=0.0003`
- `upstream_lr=0.00003`
- no oracle training and no direct operand labels

Fast gates:

| Condition | Final eval | Best dense step normal/operand/pair/calc | Final normal/operand/pair/calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLI seed `0` / effective seed `2` | `0.048828` | step `650`: `0.457031` / `0.457031` / `0.457031` / `0.457031` | `0.054688` / `0.054688` / `0.054688` / `0.070312` | `0.0` | `0.003906` | `1.0` |
| CLI seed `3` / effective seed `5` | `0.062500` | step `350`: `0.433594` / `0.433594` / `0.433594` / `0.441406` | `0.050781` / `0.050781` / `0.050781` / `0.082031` | `0.0` | `0.007812` | `1.0` |

Parameter deltas versus Stage 0B:

| Condition | `calculator_hook.input_proj` L2 | upstream L2 | semantic decoder L2 |
| --- | ---: | ---: | ---: |
| seed `0` | `0.860573` | `1.60121` | `0.0` |
| seed `3` | `0.808462` | `1.48781` | `0.0` |

Selected diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-minus-true/best gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: |
| seed `0` final | `0.0391` / `0.0391` / `0.0625` | `0.0375` / `0.0375` / `0.0625` | `6.1375` / `6.1375` | `0.0391` |
| seed `0` best step `650` | `0.4297` / `0.4297` / `0.4297` | `0.4200` / `0.4200` / `0.4200` | `2.5851` / `2.5851` | `0.4062` |
| seed `3` final | `0.0508` / `0.0508` / `0.0859` | `0.0575` / `0.0575` / `0.0925` | `5.6418` / `5.6418` | `0.0469` |
| seed `3` best step `350` | `0.4336` / `0.4336` / `0.4336` | `0.4500` / `0.4500` / `0.4575` | `2.0637` / `2.0637` | `0.4141` |

Interpretation:

- This is a clean no-handoff full-model initialization smoke failure.
- Oracle-at-eval stayed `1.0`, so the fixed semantic decoder/calculator path
  remained mechanically viable.
- Answer-only training did not discover the true calculator-query protocol in
  either allowed seed. The best checkpoints were partial and still had strongly
  positive full-enum learned-minus-true/best gaps.
- The final checkpoints drifted close to chance learned actions despite
  measurable upstream movement and frozen semantic decoder weights.
- Because Stage 1 produced no real no-handoff discovery checkpoint, the
  optional strict random-upstream branch was not run.

Recommendation:

Do not broaden into a seed/LR sweep. The next task should move to one minimal
local-target/full-enum target-prop style objective or a Gumbel-Softmax estimator
while keeping this strict identifiable setup and diagnostics.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "semantic_decoder_checkpoint_load_scope or freeze_semantic_decoder_preserves_decoder_but_not_interface"
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py summarize
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py run --jobs 2
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase5_no_handoff_upstream_discovery_smoke.py summarize
```
