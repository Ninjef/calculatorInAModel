# Work History: Bottleneck-to-Additive Anchor Decay Off-Ramp

Date: 2026-05-28

## Goal

Test whether the successful constant-anchor full-policy unfreeze can become
self-sustaining by linearly decaying the KL result-policy anchor to zero.

## Commands

`src4_add2` adapted checkpoint:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 10 --result-policy-anchor-decay-steps 200 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_decay_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor10_decay200_steps400
```

`src5_add5` adapted checkpoint:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 10 --result-policy-anchor-decay-steps 200 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_decay_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor10_decay200_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Step-200 calc | Final calc | Final anchor agreement | Final anchor accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` decay200 | `0.5925` | `0.7250` at `250` | `0.5975` | `0.0325` | `0.0975` | `0.7400` | `0.8300` | `0.5950` | `0.6200` | `0.6300` |
| `src5_add5` decay200 | `0.6750` | `0.9575` at `200` | `0.7400` | `0.0375` | `0.0800` | `0.7925` | `0.8225` | `0.3850` | `0.4300` | `0.3800` |

## Interpretation

The anchor decay schedule failed as an off-ramp. Both policies were still
usable at the shutoff point, then drifted during the anchor-free tail. The
constant-anchor partial positive therefore depends on an active policy-retention
constraint.

## Verification

No code changed in this task. Verification consisted of completed training
runs, saved metrics, and artifact extraction from `metrics.json`,
`training_curve.csv`, and `diagnostic_snapshots.csv`.
