# Work History: Bottleneck-to-Additive Anchor Threshold

Date: 2026-05-28

## Goal

Find whether constant KL anchor `0.01` is still enough to preserve transferred
calculator policies during non-bottleneck full-policy unfreeze.

## Commands

`src4_add2`, anchor `0.01`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.01 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_threshold_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_steps400
```

`src5_add5`, anchor `0.01`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.01 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_threshold_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Last calc | Anchor agree | Anchor acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_anchor0.01` | `0.7850` | `0.7850` at `400` | `0.7850` | `0.0050` | `0.1250` | `0.8200` | `0.7625` | `0.8825` | `0.7950` |
| `src5_anchor0.01` | `0.9375` | `0.9250` at `400` | `0.9250` | `0.0000` | `0.0950` | `0.9525` | `0.6425` | `0.7050` | `0.6125` |

## Interpretation

Anchor `0.01` is a mixed threshold result. It preserves causal calculator
dependence but not robust policy accuracy, especially for `src5_add5`.
Compared with anchor `0.1`, the final answer and calculator-result metrics are
both worse. Compared with no anchor, it still protects the policy partially.

## Verification

No code changed in this task. Verification used completed training runs and
artifact extraction from `metrics.json`, `training_curve.csv`, and
`diagnostic_snapshots.csv`.
