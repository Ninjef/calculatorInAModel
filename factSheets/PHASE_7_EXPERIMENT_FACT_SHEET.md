# Phase 7 Experiment Fact Sheet

## Direction

Phase 7 tests natural `0..19` result-level calculator use.

The phase starts from the Phase 6 closure:

```text
Deterministic hard-forward / soft-backward Concrete can discover and retain an
identifiable `sum_left_operand` hard calculator protocol, but natural sum-only
addition failed because the answer identifies a result group rather than a
unique operand pair.
```

Phase 7 should therefore prioritize structured joint-pair or result-space
interfaces that match the result-level information available in natural answer
loss.

## Current State After Result-Space Diagnostic

As of `2026-05-13`, both strict natural result-level action
parameterizations tried in Phase 7 are negative:

- `joint_pair_stage1_negative`: hard learned calculator-result accuracy peaked
  at `0.11`; soft true-result probability stayed near broad initial mass.
- `result_space_stage1_negative`: even a direct `0..38` result request head
  peaked at only `0.0925` hard learned calculator-result accuracy, while soft
  true-result probability moved only `0.02564 -> 0.02920`.

This means pair underidentification was real but not sufficient to explain the
natural-addition failure. The frozen product decoder/readout and full-enum
result landscape remain healthy, but deterministic hard-forward /
soft-backward Concrete answer-loss training from strict initialization is not
currently converting the result-level answer signal into a learned model-side
calculator request.

Current recommendation:

```text
Stop small sweeps of the deterministic Concrete result-level setup. Move next
to qualitatively different learning signals: policy-gradient / REINFORCE-style
calculator actions, target propagation or local boundary targets,
differentiable surrogate gradients, synthetic-gradient/direct-feedback methods,
or explicit curriculum handoffs with teacher removal.
```

Next task selected:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

Rationale:

- The fastest high-signal next branch is an answer-derived result boundary
  target, because Phase 6 local-target work showed that answer-derived boundary
  targets can teach protocols in the identifiable setting.
- The target should be computed by enumerating forced calculator result classes
  through the frozen product decoder and choosing/weighting the lowest answer
  NLL result. True sums are allowed only for post-hoc parity diagnostics.
- This is a target-propagation/local-boundary-target learning signal, not a
  small deterministic Concrete sweep.
- The decisive test is not Stage 1 target teaching alone. It is whether the
  hard learned result request survives after the boundary-target objective is
  exactly `0.0`.

## Starting Guardrail

Oracle/readout success is a wiring gate only. Phase 7 progress must be judged
by learned calculator-result behavior under the hard calculator path:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- private all-pair result accuracy;
- full-enum learned-result best fraction and learned-result gaps;
- injection-zero and forced-random controls;
- semantic decoder movement exactly `0.0`;
- auxiliary/direct operand supervision exactly `0.0`;
- all discovery-specific objective weights exactly `0.0` for retention claims.

Exact true operand-pair recovery is diagnostic only in natural sum-only
addition, because many valid calculator calls share the same correct result.

The natural `0..19` product decoder/readout is no longer an open question. It
has repeatedly passed oracle/readout and full-enum result-landscape checks.
Future Phase 7 work should not present decoder usability, oracle-at-eval
success, forced-true result success, or injection wiring as new knowledge. Run
those checks only after code or checkpoint changes that could break the path,
and label them as regression checks only.

## First Recommended Track

Start with the Phase 7 overarching plan:

```text
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
```

The first task should gate or implement a natural joint-pair result-group
deterministic Concrete bridge before attempting larger operand ranges.

## 2026-05-12 Joint-Pair Result-Group Bridge Implementation Gate

Task:

```text
aiAgentProjectTasks/completed/phase7/2026-05-12-phase-7-first-task-Natural-joint-pair-result-group-bridge-gate.md
```

Claim tested:

```text
The natural sum-only interface can use a joint 20 x 20 pair policy whose hard
forward calculator call is trained by soft backward mass grouped by calculator
result, without true operand labels or oracle operands.
```

Code changes:

- Enabled `calculator_estimator=gumbel_concrete_interface` with
  `calculator_action_head=joint_pair`.
- Enabled `calculator_read_position=operand_spans` for `joint_pair` and sized
  `pair_proj` as `2 * calculator_read_span_width * n_embd -> V^2`.
- Added joint-pair hard-forward / soft-backward result-group relaxation:
  `p_result[s] = sum_{a+b=s} p_pair[a,b]`, with hard argmax pair used for the
  forward calculator result.
- Extended relaxed calculator metrics for joint-pair policies with pair/result
  entropy, effective result count, hard pair exact, and hard calculator-result
  accuracy.
- Preserved existing independent-head relaxed behavior and existing joint
  full-enum behavior.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests -q
```

Results:

```text
tests/test_model.py: 74 passed
tests/: 83 passed
```

Phase 7 CLI smoke run:

```text
runs/2026-05-12_phase7_joint_pair_result_group_bridge_gate/stage0_cli_smoke/2026-05-12_184116_723657_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal2-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_bottleneck_mode=answer_decoder`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- `oracle_train=false`, `oracle_warmup_steps=0`
- `answer_loss_weight=1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.pair_proj` only (`26,000`)

One-step CLI smoke summary:

| Step | Answer loss | Hard pair exact | Hard result accuracy | Pair entropy | Effective pairs | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `8.3140` | `0.0000` | `0.0078` | `5.9915` | `399.9993` | `3.4932` | `32.8908` |
| `1` | `7.2661` | `0.0000` | `0.0234` | `5.9912` | `399.8765` | `3.4913` | `32.8277` |

Gradient gate on a fixed 128-sample natural batch:

| Metric | Value |
| --- | ---: |
| answer loss | `7.6510` |
| pair-proj gradient L2 | `0.04198` |
| pair-proj one-step delta L2 | `4.8305` |
| input-proj gradient L2 | `0.0` |
| semantic decoder gradient L2 | `0.0` |
| upstream gradient L2 | `0.0` |
| semantic output-proj delta L2 | `0.0` |
| `pair_proj.weight` shape | `[400, 64]` |
| initial hard result accuracy | `0.0234` |

Interpretation:

- The implementation gate passed: answer loss sends nonzero gradient through
  result-group mass into `calculator_hook.pair_proj`, while semantic decoder
  and upstream parameters remain frozen.
- This is not a learned-interface success claim yet. The smoke run is only a
  one-step wiring/gradient gate; hard learned result accuracy remains near
  chance as expected from strict initialization.

Recommendation:

Proceed to the Phase 7 Stage 1 natural decoder/full-enum landscape regression
gate, then run the seed-2 strict joint-pair bridge only if the product decoder
and result landscape gates still pass.

## 2026-05-12 Joint-Pair Stage 1 Result Discovery

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-7-second-task-Natural-joint-pair-stage1-result-discovery-and-retention-gate.md
```

Claim tested:

```text
Can answer loss train a natural 20 x 20 joint-pair calculator-query policy to
produce hard calculator calls with correct results, without true operand
labels, oracle operands, hard-best CE, expected-loss enumeration, or semantic
decoder movement?
```

Code changes:

- Added relaxed joint-pair result metrics to `scripts/overfit_one_batch.py`:
  `relaxed_calculator_true_result_probability`,
  `relaxed_calculator_argmax_result_accuracy`, and
  `relaxed_calculator_top3_result_accuracy`.
- Mirrored the same soft result metrics for independent relaxed policies so
  training curves keep a consistent schema.
- Added focused coverage in `tests/test_model.py` for soft-result versus
  hard-result metric reporting.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
75 passed
```

Stage 0 gates:

| Gate | Result |
| --- | ---: |
| oracle/readout exact | `1.000` |
| oracle/readout injection-zero | `0.055` |
| oracle/readout forced-random | `0.0225` |
| full-enum best result group true sum | `1.000` |
| soft target true result group probability | `0.99994` |
| soft target true pair probability | `0.09749` |

Stage 1 run:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_action_head=joint_pair`
- `calculator_estimator=gumbel_concrete_interface`
- `calculator_read_position=operand_spans`, span width `2`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`
- answer loss `1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.pair_proj` only (`26,000`)

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Pair entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0250` | `0.03364` | `0.0650` | `0.1675` | `3.4932` | `5.9915` |
| `150` | `0.0900` | `0.03447` | `0.0475` | `0.1625` | `3.4879` | `5.9906` |
| `300` | `0.0300` | `0.03456` | `0.0600` | `0.1675` | `3.4829` | `5.9887` |
| `450` | `0.1100` | `0.03383` | `0.0475` | `0.1475` | `3.4702` | `5.9821` |
| `600` | `0.0525` | `0.03643` | `0.0325` | `0.1475` | `3.4213` | `5.9542` |

Selected checkpoint:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_weights.pt
```

Selection reason:

- Best relaxed hard learned calculator-result accuracy in the Stage 1 curve:
  `0.1100`.
- This is below the near-pass threshold and near the old natural
  independent-head negative range.

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.1275` |
| canonical calculator result accuracy | `0.1275` |
| canonical result-equivalent pair accuracy | `0.1275` |
| canonical pair exact | `0.0125` |
| injection-zero exact | `0.055` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.000` |
| full-enum learned-result best fraction | `0.1125` |
| full-enum learned result matches true sum | `0.1125` |
| mean learned-result minus best-result gap | `5.5218` |
| full-enum best result group true sum | `1.000` |
| full-enum true result group probability | `0.99994` |
| full-enum true pair probability | `0.09749` |

Parameter movement from step `0` to selected step `450`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.pair_proj` | `38.0941` | `2.2904` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| upstream encoder | `0.0` | `0.0` | `0/29` |

Interpretation:

- Label: `joint_pair_stage1_negative`.
- The product decoder and result landscape are healthy, but strict seed-2
  joint-pair result-group bridge training did not discover a useful hard
  natural calculator-result protocol.
- The new metrics distinguish this from a soft-positive/hard-handoff failure:
  soft true-result probability stayed near the broad initial result mass
  (`~0.034` to `0.036`), while hard result accuracy peaked at only `0.11`.
- Retention, replication seeds `4`/`5`, upstream-open training, and
  `operand_max=99` were not run because Stage 1 did not pass or near-pass.

Recommendation:

Move next to Track B result-space interface or Track C canonical symmetry
breaker. Do not run Stage 2 retention or seed replication from this checkpoint.

## 2026-05-13 Result-Space Interface Diagnostic

Task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-third-task-Natural-result-space-interface-diagnostic.md
```

Claim tested:

```text
Can natural answer loss train a frozen-upstream model-side `0..38`
calculator-result request when the action space exactly matches the result
class identified by the answer target?
```

Code changes:

- Added `calculator_action_head=result_space`.
- Added `calculator_hook.result_proj`, mapping paired operand-span read
  representations to `calculator_result_vocab_size` logits.
- Hard forward picks `result_pred=argmax(result_logits)` and maps it to a
  deterministic valid canonical query:
  `a=min(result, operand_max)`, `b=result-a`.
- Deterministic Concrete backward uses a soft result distribution directly over
  `0..38`, with hard-forward / soft-backward calculator-output signal through
  the frozen semantic decoder.
- Added trace fields for result confidence and result entropy.
- Extended relaxed metrics and full-enum diagnostics for result-space heads.
- Added focused tests for canonical mapping coverage, result-proj gradients,
  frozen semantic/upstream gradients, relaxed metrics, and CLI/model validation.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
79 passed
```

Stage 1 run:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`,
  `calculator_result_vocab_size=39`
- `answer_format=sum`, `calculator_output_format=sum`
- `calculator_action_head=result_space`
- `calculator_estimator=gumbel_concrete_interface`
- `calculator_read_position=operand_spans`, span width `2`
- `answer_decoder_interaction=product`
- `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`
- `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`
- answer loss `1.0`
- aux/adaptive/local/expected/relaxed-entropy/anchor weights all `0.0`
- trainable parameters: `calculator_hook.result_proj` only (`2,535`)

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0075` | `0.02564` | `0.0075` | `0.0850` | `3.6636` | `38.9999` |
| `150` | `0.0675` | `0.02613` | `0.0675` | `0.1650` | `3.6627` | `38.9645` |
| `300` | `0.0325` | `0.02641` | `0.0325` | `0.1575` | `3.6602` | `38.8684` |
| `450` | `0.0425` | `0.02733` | `0.0425` | `0.1625` | `3.6523` | `38.5648` |
| `600` | `0.0925` | `0.02920` | `0.0925` | `0.1750` | `3.6163` | `37.2100` |

Selected checkpoint:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_weights.pt
```

Selection reason:

- Best hard learned calculator-result accuracy in the Stage 1 curve:
  `0.0925`.
- This is below the near-pass threshold and near the strict joint-pair
  negative range.

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.0975` |
| canonical calculator result accuracy | `0.0975` |
| canonical result-equivalent pair accuracy | `0.0975` |
| canonical pair exact | `0.0100` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| mean result confidence | `0.03018` |
| mean result entropy | `3.6502` |
| full-enum learned-result best fraction | `0.0850` |
| full-enum learned result matches true sum | `0.0850` |
| mean learned-result minus best-result gap | `4.7702` |
| best result group matches true sum | `1.0000` |
| mean soft target true result group probability | `0.99994` |
| mean soft target true pair probability | `0.09749` |

Parameter movement from step `0` to selected step `600`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.result_proj` | `18.0823` | `2.8812` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| upstream encoder | `0.0` | `0.0` | `0/29` |

Interpretation:

- Label: `result_space_stage1_negative`.
- The natural product decoder/readout path remains healthy as a wiring
  regression (`oracle-at-eval=1.0`), and the full-enum landscape remains
  result-sharp.
- Strict frozen-upstream result-space training did not learn a useful hard
  calculator-result request. Soft true-result probability rose only from
  `0.02564` to `0.02920`, while effective results remained broad at `37.21`.
- This is not a soft-positive/hard-handoff case and not a retention candidate.

Recommendation:

Do not run Stage 2 retention, seed replication, Track C canonical-query
symmetry breaking, or operand-range scaling from this checkpoint. The next work
should move to qualitatively different learning signals: policy-gradient /
REINFORCE-style calculator actions, target propagation or local boundary
targets, differentiable surrogate gradients, synthetic-gradient/direct-feedback
methods, or explicit curriculum handoffs with teacher removal.

## 2026-05-13 Result-Space Boundary-Target Learning Signal

Task:

```text
aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

Claim tested:

```text
Can an answer-derived boundary target over calculator result classes teach a
natural 0..19 model-side result request, with true sums used only for
diagnostics and parity checks?
```

Code changes:

- Added explicit result-boundary target training flags to
  `scripts/overfit_one_batch.py`:
  `--result-boundary-target-loss-weight`,
  `--result-boundary-target-mode`,
  `--result-boundary-target-temperature`,
  `--result-boundary-target-min-probability-floor`, and
  `--result-boundary-target-chunk-size`.
- Added forced-result-class scoring over result classes `0..38`, using the
  frozen product answer decoder to compute answer NLL for each candidate.
- Added hard-best result CE and soft-result CE/KL targets on
  `calculator_hook.result_proj`; target construction does not use true operands
  or true sums.
- Logged result-boundary target settings and metrics separately from prior
  operand-pair local-target metrics.
- Added tests for lowest-NLL target selection, result-proj gradient flow,
  frozen semantic/upstream gradients, parity with direct true-sum CE only after
  target construction, and CLI validation.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
83 passed
```

Stage 0 boundary-target parity gate from the Phase 6 product checkpoint:

| Metric | Value |
| --- | ---: |
| hard-best result equals true sum | `1.0000` |
| tie-aware true-result best fraction | `1.0000` |
| soft target true-result probability | `0.99989` |
| target entropy | `0.00106` |
| effective result count | `1.0011` |
| initial hard learned result accuracy | `0.0250` |
| result-proj gradient L2 | `0.10210` |
| semantic decoder gradient/delta L2 | `0.0 / 0.0` |
| upstream gradient/delta L2 | `0.0 / 0.0` |
| trainable group | `calculator_hook.result_proj` only |

Stage 1 primary run:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `result_boundary_target_temperature=0.25`
- `input_proj_lr=0.03`
- `steps=300`
- frozen semantic decoder and frozen upstream encoder
- trainable parameters: `calculator_hook.result_proj` only (`2,535`)

Primary curve summary:

| Step | Boundary loss | Learned result acc | Learned-best fraction | Learned-minus-best gap | Result entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `3.6638` | `0.0075` | `0.0075` | `7.5011` | `3.6637` |
| `75` | `3.2669` | `0.0600` | `0.0600` | `7.0230` | `3.4690` |
| `150` | `3.1388` | `0.0925` | `0.0925` | `6.7583` | `3.4150` |
| `175` | `3.1153` | `0.1150` | `0.1150` | `6.6698` | `3.3820` |
| `300` | `2.9622` | `0.0700` | `0.0700` | `6.7682` | `3.3110` |

Because the primary run did not reach `0.70`, the single allowed optimization
rescue was run with `input_proj_lr=0.01` and `steps=600`:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best_lr001_rescue/2026-05-13_072601_947478_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Rescue result:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.0900` at step `250` |
| final hard learned calculator-result accuracy | `0.0750` |
| final eval exact | `0.0650` |

Selected Stage 1 checkpoint:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00175_weights.pt
```

Selection reason: primary step `175` had the best hard learned
calculator-result accuracy across primary and rescue runs (`0.1150`).

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.0850` |
| canonical calculator result accuracy | `0.0850` |
| canonical result-equivalent pair accuracy | `0.0850` |
| canonical pair exact | `0.0125` |
| canonical injection-zero exact | `0.0550` |
| canonical forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| mean result confidence | `0.06118` |
| mean result entropy | `3.3898` |
| full-enum learned-result best fraction | `0.0850` |
| full-enum learned result matches true sum | `0.0850` |
| mean learned-result minus best-result gap | `6.8508` |
| best result group matches true sum | `1.0000` |
| mean soft target true result-group probability | `0.99995` |
| mean effective result count | `1.0009` |

Parameter movement from Stage 1 step `0` to selected step `175`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.result_proj` | `113.5894` | `5.1529` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| upstream encoder | `0.0` | `0.0` | `0/29` |
| other interface groups | `0.0` | `0.0` | `0/2` |

Interpretation:

- Label: `result_boundary_target_stage1_negative`.
- The result-boundary target itself is valid and sharp: Stage 0 hard-best and
  tie-aware true-result gates were both `1.0`, and selected-checkpoint
  full-enum diagnostics still show the true result group as best with
  probability `0.99995`.
- Despite that, the frozen operand-span features plus `result_proj` did not
  learn a useful hard result request. The best hard result accuracy was only
  `0.1150`, and the single allowed LR rescue reached only `0.0900`.
- Stage 2 target-off retention was skipped because Stage 1 did not pass or
  near-pass.

Recommendation:

Do not replicate this branch or run Stage 2 from these checkpoints. The next
task should pivot to a different signal family or capacity/feature diagnosis:
multi-sample policy gradient with per-prompt baselines, surrogate gradients,
direct feedback alignment, or a direct separability test of whether frozen
operand-span representations can linearly predict the answer-derived result
target.
