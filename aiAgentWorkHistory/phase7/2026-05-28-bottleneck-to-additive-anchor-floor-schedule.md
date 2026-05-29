# Work History: Bottleneck-to-Additive Anchor Floor Schedule

Date: 2026-05-28

## Goal

Add a nonzero floor to the result-policy anchor schedule and test whether
anchor `1.0 -> 0.1` over `200` steps preserves non-bottleneck calculator use.

## Code Change

- Added `--result-policy-anchor-floor`.
- Added `result_policy_anchor_weight_schedule`.
- Logged the configured floor and final scheduled anchor weight.
- Added a focused unit test for the floored schedule.

## Commands

Focused verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "result_policy_anchor or adaptive_interface_weight_schedule or aux_operand_weight"
```

`src4_add2`, anchor `1 -> 0.1`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 1 --result-policy-anchor-decay-steps 200 --result-policy-anchor-floor 0.1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_floor_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor1_decay200_floor0.1_steps400
```

`src5_add5`, anchor `1 -> 0.1`:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 1 --result-policy-anchor-decay-steps 200 --result-policy-anchor-floor 0.1 --result-policy-anchor-mode kl --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_floor_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor1_decay200_floor0.1_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Last calc | Final anchor weight | Anchor agree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_floor0.1` | `0.7925` | `0.7725` at `200` | `0.7500` | `0.0250` | `0.1000` | `0.8000` | `0.8175` | `0.1000` | `0.9225` |
| `src5_floor0.1` | `0.9775` | `0.9650` at `350` | `0.9600` | `0.0075` | `0.0550` | `0.9300` | `0.7800` | `0.1000` | `0.8975` |

## Interpretation

The floored schedule is a partial positive. It avoids the failed zero-off-ramp
collapse and confirms the `0.1` anchor region is a useful lightweight retention
floor. It does not beat constant `0.1`, so future work should test adaptive or
calculator-accuracy-gated retention rather than more fixed floors.
