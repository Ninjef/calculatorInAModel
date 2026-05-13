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
