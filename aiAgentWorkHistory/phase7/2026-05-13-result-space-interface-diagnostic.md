# Phase 7 Result-Space Interface Diagnostic

## Task

```text
aiAgentProjectTasks/2026-05-13-phase-7-third-task-Natural-result-space-interface-diagnostic.md
```

## Claim

Strict natural `0..19` answer loss was tested as a teacher for a model-side
`0..38` calculator-result request. The head predicts the calculator result
class directly, maps that result to a deterministic canonical valid query, and
feeds the real calculator result through the frozen product decoder.

## Code Changes

- Added `calculator_action_head=result_space` in `src/model.py`.
- Added `calculator_hook.result_proj`, using the paired calculator read
  representation to predict `calculator_result_vocab_size` result logits.
- Added canonical mapping from predicted result class to a valid query:
  `a=min(result, operand_max)`, `b=result-a`.
- Added deterministic hard-forward / soft-backward Concrete over result
  classes:
  `hard_one_hot(result_pred).detach() + soft_result_probs - soft_result_probs.detach()`.
- Added result confidence and result entropy trace fields and diagnostic CSV
  output.
- Extended `scripts/overfit_one_batch.py` result-space metric logging:
  `relaxed_calculator_true_result_probability`,
  `relaxed_calculator_argmax_result_accuracy`,
  `relaxed_calculator_top3_result_accuracy`,
  `relaxed_calculator_hard_learned_calc_accuracy`,
  `relaxed_calculator_result_entropy`, and
  `relaxed_calculator_effective_results`.
- Extended `scripts/run_full_enum_action_loss_diagnostic.py` to decode learned
  result-space requests.
- Added focused tests in `tests/test_model.py`.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
79 passed
```

## Stage 1 Command

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 600 --batch-size 400 --eval-samples 400 --operand-max 19 --calculator-operand-vocab-size 20 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --answer-format sum --calculator-output-format sum --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --calculator-estimator gumbel_concrete_interface --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 0.5 --relaxed-calculator-temperature-decay-steps 600 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --snapshot-every 25 --snapshot-samples 400 --checkpoint-every 25 --log-every 25 --run-root runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary
```

Run path:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Final eval exact:

```text
0.080
```

## Stage 1 Selection

Selected checkpoint:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_weights.pt
```

Reason: step `600` had the best hard learned calculator-result accuracy in the
training curve, `0.0925`.

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0075` | `0.02564` | `0.0075` | `0.0850` | `3.6636` | `38.9999` |
| `150` | `0.0675` | `0.02613` | `0.0675` | `0.1650` | `3.6627` | `38.9645` |
| `300` | `0.0325` | `0.02641` | `0.0325` | `0.1575` | `3.6602` | `38.8684` |
| `450` | `0.0425` | `0.02733` | `0.0425` | `0.1625` | `3.6523` | `38.5648` |
| `600` | `0.0925` | `0.02920` | `0.0925` | `0.1750` | `3.6163` | `37.2100` |

Maxima:

| Metric | Best |
| --- | ---: |
| hard learned calculator-result accuracy | `0.0925` at step `600` |
| soft true-result probability | `0.02920` at step `600` |
| soft argmax result accuracy | `0.0925` at step `600` |
| top-3 result accuracy | `0.1750` at step `600` |

## Selected Checkpoint Diagnostics

Canonical diagnostic command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_weights.pt --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --output-dir runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_canonical_diagnostic
```

Full-enum diagnostic command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_weights.pt --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary/2026-05-12_203621_038904_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00600_full_enum_diagnostic
```

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.0975` |
| canonical calculator result accuracy | `0.0975` |
| canonical result-equivalent pair accuracy | `0.0975` |
| canonical pair exact | `0.0100` |
| canonical injection-zero exact | `0.0550` |
| canonical forced-random exact | `0.0225` |
| canonical oracle-at-eval exact | `1.0000` |
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

Final objective weights:

| Objective | Weight |
| --- | ---: |
| answer loss | `1.0` |
| auxiliary operand loss | `0.0` |
| adaptive interface loss | `0.0` |
| local target loss | `0.0` |
| expected answer loss | `0.0` |
| relaxed entropy | `0.0` |
| input projection anchor | `0.0` |

Trainable parameter groups:

```text
calculator_hook.result_proj only, 2,535 parameters
```

## Interpretation

Label:

```text
result_space_stage1_negative
```

The Stage 1 result-space branch did not discover a useful hard result request.
Hard calculator-result accuracy peaked at only `0.0925`, soft true-result
probability moved only from `0.02564` to `0.02920`, and the result distribution
remained broad with `37.21` effective results at the selected checkpoint.

The full-enum landscape remained result-sharp (`true_result_best_fraction=1.0`,
soft target true result-group probability `0.99994`), so the negative is not
explained by decoder/readout failure. The oracle-at-eval result (`1.0`) is a
wiring regression check only.

This is not a retention candidate. Stage 2 retention, seeds `4`/`5`, Track C
canonical-query symmetry breaking, upstream-open training, and `operand_max=99`
were intentionally skipped.

## Recommendation

Move next to qualitatively different learning signals: policy-gradient /
REINFORCE-style calculator actions, target-propagation or local boundary
targets, differentiable surrogate-gradient approaches, synthetic-gradient or
direct-feedback methods, or explicit curriculum handoffs with teacher removal.
Do not run small schedule sweeps of this result-space setup.
