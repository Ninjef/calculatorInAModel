# 2026-05-29 - Forced-Loss Adaptive Recovery Trigger

## Question

Can the scheduled-source late recovery phase be triggered by the forced-true
additive loss instead of raw source argmax accuracy?

## Motivation

The raw `result_policy_argmax_result_accuracy >= 0.65` trigger did not fire on
fresh seed 17, ending at `0.6100` source eval and `0.6825` trusted 600-step
handoff. The same seed had low `additive_forced_true_loss` by step 500, so a
geometry/objective maturity signal might trigger the low-LR/lower-aux recovery
without waiting for source argmax accuracy.

## Runs

Source:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 631 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 0 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator direct_feedback_alignment --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --result-policy-entropy-weight 0.05 --result-policy-batch-diversity-weight 0.1 --result-policy-improvement-assignment-weight 10 --result-policy-stabilization-temperature 1 --additive-forced-true-loss-weight 0.5 --additive-forced-true-start-step 50 --late-source-recovery-lr-multiplier 0.1 --late-source-recovery-additive-forced-true-loss-weight 0.1 --late-source-recovery-trigger-metric additive_forced_true_loss --late-source-recovery-trigger-threshold 0.05 --late-source-recovery-trigger-mode below --late-source-recovery-min-step 500 --freeze-semantic-decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_adaptive_forcedloss005_steps631_cpu --device cpu
```

Handoff:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --seed 17 --steps 600 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 1 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode none --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_adaptive_forcedloss005_steps631_cpu/2026-05-29_152911_351417_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed19/final_weights.pt --semantic-decoder-checkpoint-load-scope compatible_model --freeze-semantic-decoder --freeze-calculator-policy --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --run-root runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_handoff600_from_adaptive_forcedloss005_cpu --device cpu
```

## Results

| Branch | Source final | Handoff final | Step-600 normal | Injection-zero | Forced-random | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw source-accuracy trigger, no fire | `0.6100` | `0.6825` | `0.6925` | `0.0400` | `0.0500` | `0.6075` |
| forced-loss trigger | `0.7225` | `0.7625` | `0.7825` | `0.0450` | `0.0325` | `0.7350` |
| fixed step-600 control | `0.7450` | `0.7675` | `0.7850` | `0.0500` | `0.0375` | `0.7350` |

The forced-loss trigger fired at step `500`, when
`additive_forced_true_loss=0.0456205`. Final metrics reported
`final_late_source_recovery_active=true`, final LR multiplier `0.1`, and final
forced-true effective weight `0.1`.

## Interpretation

Forced-true loss is a better one-metric transition signal than raw source
argmax accuracy on the hard seed 17. It essentially matched the fixed step-600
control while avoiding the raw source-accuracy trigger failure.

It did not clear the high gate. Do not rerun this same threshold as novelty;
next useful transition tests should use smoothing/patience or a conjunction of
source and geometry signals, or else move back toward scalable assignment.
