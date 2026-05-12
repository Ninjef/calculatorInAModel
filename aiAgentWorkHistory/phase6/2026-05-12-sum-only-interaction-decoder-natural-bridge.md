# Sum-Only Interaction Decoder And Natural Bridge

## Task

```text
aiAgentProjectTasks/2026-05-12-phase-6-tenth-task-Sum-only-answer-decoder-interaction-and-natural-bridge.md
```

## Code Added

- Added `answer_decoder_interaction=none|product`.
- Preserved default sum-only additive behavior.
- Enabled opt-in product interaction for sum-only strict answer decoding.
- Kept `sum_left_operand` product-compatible for old checkpoints.
- Serialized the new field through training config, metrics, diagnostics, and
  checkpoint model config.
- Updated `scripts/run_phase6_sum_only_semantic_decoder_gate.py` for the new
  interaction gate run root and narrow product/fallback ladder.
- Added tests for additive default, product readout behavior, invalid option
  validation, and old checkpoint loading without the new config field.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_phase6_sum_only_semantic_decoder_gate.py scripts/run_phase6_natural_sum_only_relaxed_bridge.py scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
72 passed
```

## Commands

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py stage0-candidates
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py stage1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py diagnostics
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_sum_only_semantic_decoder_gate.py summarize
```

Run root:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate
```

Primary summaries:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.json
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.md
```

## Results

### Stage 0

The additive existing decoder stayed blocked:

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `0.9300` |
| Full-enum best-result group matches true sum | `0.9075` |
| Injection-zero exact | `0.0050` |
| Forced-random exact | `0.0325` |
| Semantic decoder delta | `0.0` |

The tiny product interaction decoder passed the all-400 gate at step `500`:

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `1.0000` |
| Full-enum best-result group matches true sum | `1.0000` |
| Forced true result exact | `1.0000` |
| Injection-zero exact | `0.0425` |
| Forced-random exact | `0.0175` |
| Semantic decoder delta | `0.0` |

Selected semantic decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

### Stage 1

The natural deterministic Concrete bridge did not learn a correct result
protocol for effective seed `2`.

Best selected snapshot:

| Metric | Value |
| --- | ---: |
| Step | `225` |
| Fast normal answer exact | `0.1350` |
| Fast calculator-result accuracy | `0.1350` |
| Final eval exact | `0.126953` |
| Canonical normal / result accuracy | `0.1175 / 0.1175` |
| Full-enum learned-result best fraction | `0.1100` |
| Mean learned-result minus best-result gap | `5.5657` |
| Full-enum best-result group true sum | `1.0000` |
| Oracle-at-eval / injection-zero / forced-random | `1.0000 / 0.0550 / 0.0225` |
| Private result accuracy | `0.1100` |
| Semantic decoder delta | `0.0` |

Final objective weights:

```text
answer_loss_weight=1.0
final_aux_operand_loss_weight=0.0
final_adaptive_interface_loss_weight=0.0
final_local_target_loss_weight=0.0
final_expected_answer_loss_weight=0.0
final_relaxed_calculator_entropy_weight=0.0
final_input_proj_anchor_weight=0.0
```

## Interpretation

```text
sum_only_interaction_gate_positive
natural_deterministic_concrete_result_negative
```

The interaction-capable sum-only answer decoder clears the natural Stage 0
health gate. However, deterministic Concrete answer-loss training did not learn
a correct calculator-result protocol in the natural underidentified sum-only
task. Retention and cross-seed replication were not run because seed `2` failed
far below the result-level bridge gate.
