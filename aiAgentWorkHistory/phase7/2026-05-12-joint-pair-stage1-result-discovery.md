# Phase 7 Joint-Pair Stage 1 Result Discovery

## Task

```text
aiAgentProjectTasks/2026-05-12-phase-7-second-task-Natural-joint-pair-stage1-result-discovery-and-retention-gate.md
```

## Claim

Strict natural `0..19` answer loss was tested as a teacher for a joint
`20 x 20` calculator-query policy. The success criterion was learned hard
calculator-result accuracy, not exact true pair recovery.

## Code Changes

- Added soft result-level relaxed metrics in `scripts/overfit_one_batch.py`:
  `relaxed_calculator_true_result_probability`,
  `relaxed_calculator_argmax_result_accuracy`, and
  `relaxed_calculator_top3_result_accuracy`.
- Included result entropy/effective-result metrics for independent relaxed
  policies too, so relaxed metric rows are comparable across action heads.
- Added a focused test in `tests/test_model.py` verifying that joint-pair
  relaxed metrics can distinguish soft result mass, soft result argmax, top-3
  result coverage, and hard learned calculator-result accuracy.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
75 passed
```

## Stage 0 Commands

Oracle/readout regression gate:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --oracle --output-dir runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage0_oracle_gate
```

Full-enum result-landscape regression gate:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage0_full_enum_gate
```

## Stage 0 Results

| Gate | Result |
| --- | ---: |
| oracle/readout exact | `1.000` |
| oracle/readout injection-zero | `0.055` |
| oracle/readout forced-random | `0.0225` |
| oracle/readout oracle-at-eval | `1.000` |
| full-enum best result group matches true sum | `1.000` |
| mean soft target true result group probability | `0.99994` |
| mean soft target true pair probability | `0.09749` |

Interpretation: the product decoder/readout gate passed and the answer-loss
landscape remains result-sharp but pair-underidentified.

## Stage 1 Command

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 600 --batch-size 400 --eval-samples 400 --operand-max 19 --calculator-operand-vocab-size 20 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --answer-format sum --calculator-output-format sum --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --calculator-estimator gumbel_concrete_interface --calculator-action-head joint_pair --calculator-read-position operand_spans --calculator-read-span-width 2 --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 0.5 --relaxed-calculator-temperature-decay-steps 600 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --snapshot-every 25 --snapshot-samples 400 --checkpoint-every 25 --log-every 25 --run-root runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary
```

Run path:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2
```

Final eval exact:

```text
0.0925
```

## Stage 1 Selection

Selected checkpoint:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_weights.pt
```

Reason: step `450` had the best relaxed hard learned calculator-result
accuracy in the training curve, `0.1100`.

Training curve summary:

| Step | Hard result acc | Soft true-result prob | Soft argmax result acc | Top-3 result acc | Result entropy | Pair entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0250` | `0.03364` | `0.0650` | `0.1675` | `3.4932` | `5.9915` |
| `150` | `0.0900` | `0.03447` | `0.0475` | `0.1625` | `3.4879` | `5.9906` |
| `300` | `0.0300` | `0.03456` | `0.0600` | `0.1675` | `3.4829` | `5.9887` |
| `450` | `0.1100` | `0.03383` | `0.0475` | `0.1475` | `3.4702` | `5.9821` |
| `600` | `0.0525` | `0.03643` | `0.0325` | `0.1475` | `3.4213` | `5.9542` |

Maxima:

| Metric | Best |
| --- | ---: |
| hard learned calculator-result accuracy | `0.1100` at step `450` |
| soft true-result probability | `0.03643` at step `600` |
| soft argmax result accuracy | `0.0675` at step `550` |
| top-3 result accuracy | `0.1875` at step `75` |

## Selected Checkpoint Diagnostics

Canonical diagnostic command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_weights.pt --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --output-dir runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_canonical_diagnostic
```

Full-enum diagnostic command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_weights.pt --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary/2026-05-12_192703_156649_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay600-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00450_full_enum_diagnostic
```

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.1275` |
| canonical calculator result accuracy | `0.1275` |
| canonical result-equivalent pair accuracy | `0.1275` |
| canonical pair exact | `0.0125` |
| canonical injection-zero exact | `0.055` |
| canonical forced-random exact | `0.0225` |
| canonical oracle-at-eval exact | `1.000` |
| full-enum learned-result best fraction | `0.1125` |
| full-enum learned result matches true sum | `0.1125` |
| mean learned-result minus best-result gap | `5.5218` |
| best result group matches true sum | `1.000` |
| mean soft target true result group probability | `0.99994` |
| mean soft target true pair probability | `0.09749` |

Parameter movement from step `0` to selected step `450`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| `calculator_hook.pair_proj` | `38.0941` | `2.2904` | `2/2` |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| upstream encoder | `0.0` | `0.0` | `0/29` |

Final objective weights:

| Objective | Weight |
| --- | ---: |
| answer loss | `1.0` |
| auxiliary operand loss | `0.0` |
| adaptive interface loss | `0.0` |
| expected answer loss | `0.0` |
| relaxed entropy | `0.0` |
| input projection anchor | `0.0` |

Trainable parameter groups:

```text
calculator_hook.pair_proj only, 26,000 parameters
```

## Interpretation

Label:

```text
joint_pair_stage1_negative
```

The Stage 0 landscape still has the desired shape: true result group is sharp
and true pair mass is broad. The strict seed-2 joint-pair result-group bridge
did not convert that into either soft result learning or hard result learning.
Soft true-result probability stayed near the broad initial result mass, hard
calculator-result accuracy peaked at `0.11`, and the full-enum learned-result
gap remained large (`5.5218`).

This is not a retention candidate. Stage 2 retention, seeds `4`/`5`,
upstream-open training, and `operand_max=99` were intentionally skipped.

## Recommendation

Proceed next to Track B result-space interface or Track C canonical symmetry
breaker. Do not replicate this strict joint-pair Stage 1 branch.
