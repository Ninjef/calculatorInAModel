# 2026-05-29 Bottleneck-to-Additive Policy-Backbone Freeze

## Aim

Test the missing selective-unfreeze case between freezing the full calculator
policy and freezing only the action head.

## Code

Changed:

- `scripts/overfit_one_batch.py`
- `tests/test_model.py`

The new `--freeze-calculator-policy-backbone` option freezes embeddings and
pre-hook blocks while leaving the calculator action head trainable.

## Runs

Run root:

```text
runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_unfreeze
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- Continued from adapted weak-source frozen-policy handoff checkpoints.
- Full model checkpoint load.
- `--freeze-calculator-policy-backbone`.
- No result-policy anchor.
- LR `3e-4`, 400 steps, snapshots every 50.

Specific cells:

| Cell | Run directory |
| --- | --- |
| `src4_add2` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_freeze_policy_backbone_steps400/2026-05-28_191349_312889_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_freeze_policy_backbone_steps400/2026-05-28_191349_313014_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Final eval | Best normal | Last normal | Last inj-zero | Last forced-random | Last oracle | Last calc | Trainable groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src4_backbone_freeze` | `0.7250` | `0.7250` at `400` | `0.7250` | `0.0400` | `0.0950` | `0.7675` | `0.8200` | `calculator_hook.result_proj`, `upstream` |
| `src5_backbone_freeze` | `0.8700` | `0.8750` at `300` | `0.8725` | `0.0075` | `0.0375` | `0.8850` | `0.8025` | `calculator_hook.result_proj`, `upstream` |

## Conclusion

Freezing the policy backbone preserves calculator-result accuracy without an
anchor, unlike full low-LR unfreeze or action-head-only freezing. The answer
readout improves over the frozen-adapted weak baselines, but the result remains
below the lightweight anchored unfreeze family.

Label:

```text
bottleneck_to_additive_policy_backbone_freeze_partial
```

## Anti-Rerun Note

Do not repeat no-anchor `--freeze-calculator-policy-backbone`, LR `3e-4`,
400-step unfreeze from the same adapted `src4_add2/src5_add5` checkpoints as
novelty.
