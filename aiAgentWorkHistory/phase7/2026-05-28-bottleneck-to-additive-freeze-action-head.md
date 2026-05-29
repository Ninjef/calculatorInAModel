# Work History: Bottleneck-to-Additive Freeze Action Head

Date: 2026-05-28

## Goal

Test whether freezing only the calculator action head prevents policy collapse
during non-bottleneck unfreezing.

## Code Change

- Added `freeze_calculator_action_head_parameters`.
- Added `--freeze-calculator-action-head`.
- Added metrics/config tracking for the flag.
- Added a unit test verifying result-space `result_proj` freezes while
  surrounding model parameters remain trainable.

## Commands

Focused verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "freeze_calculator_action_head or freeze_semantic_decoder_preserves_decoder or semantic_decoder_checkpoint_load_scope"
```

`src4_add2`, freeze action head:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 2 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed4_additive_seed2_continue800_freeze_policy/2026-05-28_175222_556848_model-c-op0-19-fullgrid/model-c-2digit-seed4/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --freeze-calculator-action-head --run-root runs/2026-05-28_phase7_bottleneck_to_additive_selective_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_freeze_action_head_steps400
```

`src5_add5`, freeze action head:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --batch-size 400 --eval-samples 400 --steps 400 --snapshot-every 50 --snapshot-samples 400 --answer-loss-weight 1 --seed 5 --lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode none --calculator-output-format sum --semantic-decoder-checkpoint runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation/source_seed5_additive_seed5_continue800_freeze_policy/2026-05-28_175222_556897_model-c-op0-19-fullgrid/model-c-2digit-seed7/final_weights.pt --semantic-decoder-checkpoint-load-scope full_model --freeze-calculator-action-head --run-root runs/2026-05-28_phase7_bottleneck_to_additive_selective_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_freeze_action_head_steps400
```

## Metrics

| Run | Final eval | Best normal | Last normal | Last injection-zero | Last forced-random | Last oracle | Last calc | Trainable groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src4_freeze_action_head` | `0.5200` | `0.6500` at `0` | `0.5325` | `0.0225` | `0.1200` | `0.7100` | `0.3000` | `upstream` |
| `src5_freeze_action_head` | `0.8100` | `0.8325` at `0` | `0.7950` | `0.0225` | `0.1125` | `0.7900` | `0.2525` | `upstream` |

## Interpretation

The result is negative. Freezing `result_proj` alone does not protect the
policy because upstream representation drift can still move examples across
the fixed action-head decision boundary. Behavior-level anchoring or freezing
the entire policy path remains necessary.
