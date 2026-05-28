# 2026-05-28 Bottleneck-to-Additive Transfer Gate

## Goal

Test whether the useful bottleneck hard-assignment result policy can be handed
to an additive non-bottleneck model without losing calculator-path dependence.

## Code Changes

- Added `compatible_model` to `--semantic-decoder-checkpoint-load-scope`.
  It loads only checkpoint tensors whose keys exist in the target model and
  whose shapes match.
- Added `--freeze-calculator-policy`.
  It freezes token/position embeddings, blocks before the calculator hook, and
  the result action head while leaving `calculator_hook.output_proj`,
  post-hook blocks, `ln_f`, and `lm_head` trainable.
- Added focused tests for compatible loading into an additive result-space
  model and for the frozen-policy trainable-parameter grouping.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_gate
```

Source checkpoint:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_convergence_gate/answer0_w10_steps1600/2026-05-28_164332_598334_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

Additive configuration:

- `calculator_bottleneck_mode=none`
- `calculator_estimator=ste`
- `calculator_action_head=result_space`
- answer loss weight `1`
- no assignment target
- exact-grid natural `0..19`
- CLI seed `2`
- 800 steps

## Results

| Setup | Final eval exact | Best normal | Last injection-zero | Last forced-random | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compatible load, no freeze | `0.7825` | `0.8075` | `0.7675` | `0.7375` | `0.7825` | `0.0250` |
| compatible load, freeze policy | `0.9400` | `0.9475` | `0.0175` | `0.0500` | `0.9600` | `0.9200` |

Policy-retention trace:

| Setup | Step `0` learned calc | Step `50` learned calc | Step `800` learned calc |
| --- | ---: | ---: | ---: |
| compatible load, no freeze | `0.9125` | `0.0300` | `0.0250` |
| compatible load, freeze policy | `0.9125` | `0.8925` | `0.9200` |

## Conclusion

Label:

```text
bottleneck_to_additive_freeze_policy_handoff_partial_positive
```

Compatible loading works. Without freezing, answer-only additive training
destroys the transferred calculator policy and solves mostly through the
bypass. With the calculator policy frozen, the additive downstream path learns
to use the calculator: normal/oracle are high, learned calculator-result
accuracy stays high, and injection-zero/forced-random are near chance.

This is not yet the final scalable or non-prescriptive solution because the
policy was first learned in a bottleneck with a forced-assignment objective and
then frozen.

## Verification

Focused verification after code changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "semantic_decoder_checkpoint_load_scope or non_bottleneck_result_space_assignment"
```

Result: `2 passed, 104 deselected`.

Final verification before commit:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
git diff --check
```

Result: `106 passed`.
