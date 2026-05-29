# Work History: Bottleneck-to-Additive Reduced Anchor Strength

Date: 2026-05-28

## Goal

Measure whether the constant KL result-policy anchor can be much smaller than
the original weight `10` while still preserving calculator use during
non-bottleneck full-policy unfreeze.

## Commands

All runs used exact-grid natural `0..19`, full-model checkpoint loading, no
`--freeze-calculator-policy`, LR `3e-4`, and 400 steps.

`src4_add2`, anchor `1.0`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor1_steps400
```

`src5_add5`, anchor `1.0`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor1_steps400
```

`src4_add2`, anchor `0.1`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.1_steps400
```

`src5_add5`, anchor `0.1`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.1_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Last calc | Anchor agree | Anchor acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_anchor1` | `0.7775` | `0.7550` at `400` | `0.7550` | `0.0225` | `0.0950` | `0.8025` | `0.8050` | `0.9625` | `0.8425` |
| `src5_anchor1` | `0.9925` | `0.9825` at `400` | `0.9825` | `0.0075` | `0.0475` | `0.9350` | `0.7925` | `0.9625` | `0.8025` |
| `src4_anchor0.1` | `0.8325` | `0.8275` at `400` | `0.8275` | `0.0250` | `0.1100` | `0.8650` | `0.8075` | `0.9225` | `0.8375` |
| `src5_anchor0.1` | `0.9750` | `0.9700` at `400` | `0.9700` | `0.0000` | `0.0525` | `0.9150` | `0.7725` | `0.9075` | `0.7775` |

## Interpretation

The reduced-anchor gate is a partial positive. Anchor weights `1.0` and `0.1`
both preserved calculator use and improved final eval over frozen adapted
baselines. The active anchor still matters, especially given the failed
anchor-decay off-ramp, but it does not need to be as large as `10`.

## Verification

No code changed in this task. Verification used completed training runs and
artifact extraction from `metrics.json`, `training_curve.csv`, and
`diagnostic_snapshots.csv`.
