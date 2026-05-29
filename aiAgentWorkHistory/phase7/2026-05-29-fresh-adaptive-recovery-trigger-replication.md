# 2026-05-29 - Fresh Adaptive Recovery Trigger Replication

## Question

Does the simple source-accuracy adaptive late-source recovery trigger replicate
on a fresh scheduled source seed?

## Runs

Adaptive trigger source:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 631 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 0 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator direct_feedback_alignment --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --result-policy-entropy-weight 0.05 --result-policy-batch-diversity-weight 0.1 --result-policy-improvement-assignment-weight 10 --result-policy-stabilization-temperature 1 --additive-forced-true-loss-weight 0.5 --additive-forced-true-start-step 50 --late-source-recovery-lr-multiplier 0.1 --late-source-recovery-additive-forced-true-loss-weight 0.1 --late-source-recovery-trigger-metric result_policy_argmax_result_accuracy --late-source-recovery-trigger-threshold 0.65 --late-source-recovery-trigger-mode above --late-source-recovery-min-step 500 --freeze-semantic-decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_adaptive_acc065_steps631_cpu --device cpu
```

Adaptive-trigger handoff:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 600 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 1 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode none --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_adaptive_acc065_steps631_cpu/2026-05-29_151333_160394_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed19/final_weights.pt --semantic-decoder-checkpoint-load-scope compatible_model --freeze-semantic-decoder --freeze-calculator-policy --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_handoff600_from_adaptive_acc065_no_trigger_cpu --device cpu
```

Fixed step-600 control:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 631 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 0 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator direct_feedback_alignment --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --result-policy-entropy-weight 0.05 --result-policy-batch-diversity-weight 0.1 --result-policy-improvement-assignment-weight 10 --result-policy-stabilization-temperature 1 --additive-forced-true-loss-weight 0.5 --additive-forced-true-start-step 50 --late-source-recovery-start-step 600 --late-source-recovery-lr-multiplier 0.1 --late-source-recovery-additive-forced-true-loss-weight 0.1 --freeze-semantic-decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_fixed_step600_recovery_steps631_cpu --device cpu
```

Fixed-control handoff:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 600 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 1 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode none --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_fixed_step600_recovery_steps631_cpu/2026-05-29_151908_570045_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed19/final_weights.pt --semantic-decoder-checkpoint-load-scope compatible_model --freeze-semantic-decoder --freeze-calculator-policy --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_handoff600_from_fixed_step600_cpu --device cpu
```

## Results

| Run | Source final | Handoff final | Step-600 normal | Injection-zero | Forced-random | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive `argmax >= 0.65` | `0.6100` | `0.6825` | `0.6925` | `0.0400` | `0.0500` | `0.6075` |
| fixed step-600 recovery | `0.7450` | `0.7675` | `0.7850` | `0.0500` | `0.0375` | `0.7350` |

The adaptive trigger never fired. Its final metrics show
`late_source_recovery_trigger_step=null`, `final_late_source_recovery_active=false`,
and final forced-true weight still `0.5`.

## Interpretation

This is a fresh-seed negative for raw source-argmax thresholding. The matched
fixed step-600 control did better, so the failure is at least partly the
trigger not activating a useful late phase. The fixed control also missed the
high non-bottleneck gate, which means this seed is harder overall; still, raw
`argmax_result_accuracy >= 0.65` should not be treated as a validated adaptive
transition criterion.

Next useful work: a smoothed/patience trigger, a conjunction with
forced-true/additive geometry, or a different transition metric. Do not rerun
this exact seed-17 threshold/control pair as novelty.
