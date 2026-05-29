# Work History: Bottleneck-to-Additive Behavior-Gated Anchor

Date: 2026-05-28

## Goal

Add behavior-gated result-policy anchoring and test whether a low base anchor
can boost only when policy agreement drifts.

## Code Change

- Added `result_policy_anchor_effective_weight`.
- Added `--result-policy-anchor-gate-threshold`.
- Added `--result-policy-anchor-gate-weight`.
- Added `--result-policy-anchor-gate-metric`.
- Added training-curve columns for gate configuration, metric value, active
  state, base weight, and effective weight.
- Added focused helper test coverage.

## Commands

Focused verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "result_policy_anchor_weight_schedule or result_policy_anchor_penalizes"
```

`src4_add2`, gated anchor:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.01 --result-policy-anchor-mode kl --result-policy-anchor-gate-threshold 0.9 --result-policy-anchor-gate-weight 0.1 --result-policy-anchor-gate-metric argmax_agreement --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_gated_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_gate0.9_to0.1_steps400
```

`src5_add5`, gated anchor:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --result-policy-anchor-weight 0.01 --result-policy-anchor-mode kl --result-policy-anchor-gate-threshold 0.9 --result-policy-anchor-gate-weight 0.1 --result-policy-anchor-gate-metric argmax_agreement --run-root runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_gated_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_gate0.9_to0.1_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Last calc | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_gated` | `0.8050` | `0.8025` at `400` | `0.8025` | `0.0400` | `0.1025` | `0.8425` | `0.7700` | `4/9` | `0.0500` |
| `src5_gated` | `0.9675` | `0.9525` at `350` | `0.9450` | `0.0000` | `0.0575` | `0.9425` | `0.7700` | `8/9` | `0.0900` |

## Interpretation

Behavior gating worked mechanically. It improved over the constant `0.01`
threshold result, especially for `src5_add5`, but it did not outperform the
fixed `0.1` anchor. Future adaptive retention should use a more informative
metric, continuous weighting, or a gate tied directly to calculator-result
accuracy.
