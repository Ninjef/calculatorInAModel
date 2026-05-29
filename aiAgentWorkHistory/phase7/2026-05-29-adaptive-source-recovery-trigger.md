# 2026-05-29 - Adaptive Source Recovery Trigger

## Question

Can the late scheduled-source recovery phase be triggered from source-training
metrics instead of a fixed step?

## Code Changes

- Added adaptive late-source recovery flags to `scripts/overfit_one_batch.py`:
  `--late-source-recovery-trigger-metric`,
  `--late-source-recovery-trigger-threshold`,
  `--late-source-recovery-trigger-mode`, and
  `--late-source-recovery-min-step`.
- Supported trigger metrics:
  `result_policy_argmax_result_accuracy` and `additive_forced_true_loss`.
- Logged trigger value, trigger step, active recovery state, and effective LR.
- Fixed the adaptive path so a triggered recovery changes both optimizer LR and
  the effective forced-true auxiliary weight. Earlier exploratory runs from
  this turn switched LR only and are superseded by the fixed-override run.

## Runs

Source, corrected adaptive recovery:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 16 --steps 631 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 0 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator direct_feedback_alignment --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --result-policy-entropy-weight 0.05 --result-policy-batch-diversity-weight 0.1 --result-policy-improvement-assignment-weight 10 --result-policy-stabilization-temperature 1 --additive-forced-true-loss-weight 0.5 --additive-forced-true-start-step 50 --late-source-recovery-lr-multiplier 0.1 --late-source-recovery-additive-forced-true-loss-weight 0.1 --late-source-recovery-trigger-metric result_policy_argmax_result_accuracy --late-source-recovery-trigger-threshold 0.65 --late-source-recovery-trigger-mode above --late-source-recovery-min-step 500 --freeze-semantic-decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_auto_recovery/seed14_adaptive_acc065_fixed_override_steps631_cpu --device cpu
```

Handoff:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 16 --steps 600 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 1 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode none --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-29_phase7_scheduled_source_auto_recovery/seed14_adaptive_acc065_fixed_override_steps631_cpu/2026-05-29_150331_983432_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed18/final_weights.pt --semantic-decoder-checkpoint-load-scope compatible_model --freeze-semantic-decoder --freeze-calculator-policy --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_auto_recovery/seed14_handoff600_from_adaptive_acc065_fixed_override_cpu --device cpu
```

## Results

| Run | Final eval | Key snapshot | Controls |
| --- | ---: | --- | --- |
| adaptive source | `0.8250` | trigger step `528`; step-600 source normal/calc `0.8825`; final LR multiplier `0.1`; final forced-true weight `0.1` | step-600 zero `0.0425`, forced-random `0.0200` |
| 600-step handoff | `0.9850` | step-600 normal `0.9775`, oracle `0.9775`, learned calc `0.8425` | zero `0.1325`, forced-random `0.1325` |

Fixed step-600 automated recovery reference:

- Source final eval `0.8775`.
- Handoff final eval `0.9400`.
- Handoff controls: injection-zero `0.0800`, forced-random `0.0775`,
  learned calc `0.8725`.

## Interpretation

The simple source-accuracy trigger is a mixed-positive replacement for the
fixed transition step on this seed. It fires later than the min step but before
the fixed step-600 switch, and the resulting frozen-policy handoff beats the
fixed-step handoff final eval. The tradeoff is higher zero/random controls and
lower final source eval, so source accuracy alone is not a reliable arbiter.

Do not rerun the same seed-14 `argmax_result_accuracy >= 0.65` min-step-500
gate as novelty. Next useful tests are fresh-seed replication or a
smoothed/conjunctive trigger that preserves handoff lift while reducing
zero/random controls.

## Verification

```bash
python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. pytest tests/test_model.py -q -k 'late_source_recovery'
```
