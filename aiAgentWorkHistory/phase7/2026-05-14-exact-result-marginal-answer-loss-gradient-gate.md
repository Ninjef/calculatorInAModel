# Exact Result-Marginal Answer-Loss Gradient Gate

## Task

```text
aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
```

## Claim Tested

Does the exact expected answer-loss gradient over natural result-space actions
align with the known good result-boundary direction, or was the previous
sampled PG negative mostly finite-sample variance/control-variate weakness?

## Code Changes

- Allowed `calculator_action_head=result_space` with
  `calculator_estimator=full_enum_expected_answer_loss`.
- Added a result-space exact expected answer-loss branch in
  `scripts/overfit_one_batch.py`:
  - enumerates forced result classes `0..38`;
  - uses detached answer-NLL costs;
  - computes `mean_i sum_r p_i(r) * stopgrad(C_i(r))`;
  - records result-space expected/best/true/learned NLL, policy mass, entropy,
    effective results, hard learned-best fraction, and hard learned result
    accuracy.
- Added `--expected-answer-loss-gradient-diagnostic-only`, which computes exact
  result-marginal, sampled result-space PG, and boundary-target gradients on
  the same exact-grid batch.
- Added focused tests for result-space expected answer-loss metrics and
  gradient flow into `calculator_hook.result_proj`.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
94 passed
```

## Stage 0 Commands

Raw costs:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator full_enum_expected_answer_loss --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --exhaustive-grid-batch --answer-loss-weight 0.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --expected-answer-loss-policy-temperature 1.0 --expected-answer-loss-cost-normalization none --expected-answer-loss-entropy-weight 0.0 --expected-answer-loss-chunk-size 64 --result-boundary-target-loss-weight 0.0 --result-boundary-target-mode hard_best_result --result-boundary-target-temperature 1.0 --result-boundary-target-chunk-size 64 --input-proj-anchor-weight 0.0 --reinforce-baseline-mode leave_one_out --reinforce-num-samples-per-prompt 16 --reinforce-entropy-weight 0.0 --input-proj-lr 0.01 --upstream-lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --steps 0 --batch-size 400 --eval-samples 64 --seed 2 --run-root runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate --expected-answer-loss-gradient-diagnostic-only
```

Z-score costs:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator full_enum_expected_answer_loss --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --exhaustive-grid-batch --answer-loss-weight 0.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --expected-answer-loss-policy-temperature 1.0 --expected-answer-loss-cost-normalization zscore --expected-answer-loss-entropy-weight 0.0 --expected-answer-loss-chunk-size 64 --result-boundary-target-loss-weight 0.0 --result-boundary-target-mode hard_best_result --result-boundary-target-temperature 1.0 --result-boundary-target-chunk-size 64 --input-proj-anchor-weight 0.0 --reinforce-baseline-mode leave_one_out --reinforce-num-samples-per-prompt 16 --reinforce-entropy-weight 0.0 --input-proj-lr 0.01 --upstream-lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --steps 0 --batch-size 400 --eval-samples 64 --seed 2 --run-root runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate_zscore --expected-answer-loss-gradient-diagnostic-only
```

## Results

Raw-cost artifact:

```text
runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate/2026-05-14_093048_129116_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-expansgraddiag-answer_decoder-adec-product/model-c-2digit-seed4/expected_answer_loss_gradient_diagnostic_summary.json
```

| Metric | Value |
| --- | ---: |
| exact result-proj grad L2 | `0.1465` |
| exact upstream grad L2 | `0.0549` |
| exact semantic decoder grad L2 | `0.0` |
| exact-vs-boundary result-proj cosine | `-0.0978` |
| exact-vs-boundary upstream cosine | `-0.1231` |
| sampled PG-vs-exact result-proj cosine | `0.9577` |
| sampled PG-vs-exact upstream cosine | `0.9736` |
| sampled PG-vs-boundary result/upstream cosine | `-0.0945 / -0.1108` |
| exact expected answer-loss objective | `7.8521` |
| expected-minus-best NLL gap | `7.8517` |
| hard learned result accuracy | `0.0225` |

Z-score artifact:

```text
runs/2026-05-14_phase7_exact_result_marginal_answer_loss_gradient_gate/stage0_gradient_gate_zscore/2026-05-14_093119_897295_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-zscore-expansgraddiag-answer_decoder-adec-product/model-c-2digit-seed4/expected_answer_loss_gradient_diagnostic_summary.json
```

Z-score normalization improved the result-proj cosine to `0.0764`, but the
upstream cosine remained non-positive at `-0.0007`, so the strict upstream-open
gate still failed.

## Decision

```text
result_space_expected_answer_loss_alignment_negative
```

Stage 1 exact-marginal training was skipped.

## Interpretation

The previous sampled result-space PG negative was not mainly a finite-sample
variance artifact. With raw costs, sampled PG is strongly aligned with the
exact expected-cost gradient, and the exact expected-cost gradient itself is
anti-aligned with the boundary-target ceiling. Detached z-score normalization
does not rescue the strict upstream-open gate.

Do not spend long-run budget on raw exact expected-cost training, vanilla
result-space PG, or learned-baseline methods that merely estimate the same raw
expected-cost gradient. The next useful branch should change the learning
signal itself: surrogate/shadow-calculator gradients, synthetic
gradients/direct feedback alignment, stricter decoder-phase bottlenecks, or
another estimator that first passes the same three-way gradient gate.
