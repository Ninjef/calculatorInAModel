# Result-Space Boundary-Target Learning Signal

## Task

```text
aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

## Claim Tested

Can an answer-derived boundary target over calculator result classes teach a
natural `0..19` model-side result request, with true sums used only for
diagnostics/parity and the semantic decoder/upstream encoder frozen?

## Code Changes

- Added result-boundary target flags to `scripts/overfit_one_batch.py`:
  `--result-boundary-target-loss-weight`,
  `--result-boundary-target-mode`,
  `--result-boundary-target-temperature`,
  `--result-boundary-target-min-probability-floor`, and
  `--result-boundary-target-chunk-size`.
- Added forced result-class scoring over `0..38` through the frozen product
  answer decoder.
- Added hard-best result CE and soft-result CE/KL targets for
  `calculator_hook.result_proj`.
- Logged result-boundary target metrics under explicit names to keep them
  separate from the older operand-pair local-target metrics.
- Added focused tests for target construction, gradient flow, frozen groups,
  parity with direct true-sum CE after target construction, and CLI validation.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
83 passed
```

## Stage 0 Parity Gate

Checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Fixed natural `0..19` batch, `400` examples.

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

Gate passed, so Stage 1 proceeded.

## Stage 1 Primary

Run:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

```text
answer_loss_weight=0.0
result_boundary_target_loss_weight=1.0
result_boundary_target_mode=hard_best_result
result_boundary_target_temperature=0.25
result_boundary_target_min_probability_floor=0.0
result_boundary_target_chunk_size=64
input_proj_lr=0.03
steps=300
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.result_proj only
```

Best primary checkpoint:

```text
checkpoint_snapshots/step_00175_weights.pt
```

Primary curve:

| Step | Boundary loss | Hard result acc | Learned-best | Learned-minus-best gap | Result entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `3.6638` | `0.0075` | `0.0075` | `7.5011` | `3.6637` |
| `75` | `3.2669` | `0.0600` | `0.0600` | `7.0230` | `3.4690` |
| `150` | `3.1388` | `0.0925` | `0.0925` | `6.7583` | `3.4150` |
| `175` | `3.1153` | `0.1150` | `0.1150` | `6.6698` | `3.3820` |
| `300` | `2.9622` | `0.0700` | `0.0700` | `6.7682` | `3.3110` |

The primary run failed the `0.70` hard result accuracy threshold.

## LR Rescue

The single allowed rescue was run with `input_proj_lr=0.01`, `steps=600`, same
target settings:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best_lr001_rescue/2026-05-13_072601_947478_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.0900` at step `250` |
| final hard learned calculator-result accuracy | `0.0750` |
| final eval exact | `0.0650` |

The rescue also failed.

## Selected Checkpoint Diagnostics

Selected checkpoint:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best/2026-05-13_072413_688763_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00175_weights.pt
```

Canonical diagnostics:

| Diagnostic | Value |
| --- | ---: |
| normal exact | `0.0850` |
| calculator result accuracy | `0.0850` |
| result-equivalent pair accuracy | `0.0850` |
| pair exact | `0.0125` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| mean result confidence | `0.06118` |
| mean result entropy | `3.3898` |

Full-enum diagnostics:

| Diagnostic | Value |
| --- | ---: |
| learned-result best fraction | `0.0850` |
| learned result matches true sum | `0.0850` |
| mean learned-result minus best-result gap | `6.8508` |
| true result group best fraction | `1.0000` |
| tie-aware true best fraction | `1.0000` |
| mean soft target true result-group probability | `0.99995` |
| mean effective result count | `1.0009` |

Parameter movement from Stage 1 step `0` to selected step `175`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.result_proj` | `113.5894` | `5.1529` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| upstream encoder | `0.0` | `0.0` | `0/29` |
| other interface groups | `0.0` | `0.0` | `0/2` |

## Interpretation

Label:

```text
result_boundary_target_stage1_negative
```

The result-boundary target construction is valid and sharp: the forced-result
answer landscape identifies the true result class essentially exactly. However,
the frozen operand-span representation plus `calculator_hook.result_proj` did
not learn a useful hard result request under the primary recipe or the single
allowed LR rescue.

Stage 2 target-off retention was skipped because Stage 1 did not pass or
near-pass.

Recommended next direction: do not replicate this branch. Pivot to a different
signal family or a capacity/feature diagnosis, such as multi-sample policy
gradient with per-prompt baselines, surrogate gradients, direct feedback
alignment, or direct separability of the frozen operand-span representations
against the answer-derived result target.
