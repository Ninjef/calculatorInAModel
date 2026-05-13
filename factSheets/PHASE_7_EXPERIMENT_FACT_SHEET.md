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
