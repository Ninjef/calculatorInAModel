# 2026-05-29 Policy-Backbone Freeze Plus Tiny Anchor

## Aim

Test whether a tiny result-policy anchor adds value when the calculator policy
backbone is already frozen.

## Runs

Run root:

```text
runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_tiny_anchor
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- Continued from adapted weak-source frozen-policy handoff checkpoints.
- Full model checkpoint load.
- `--freeze-calculator-policy-backbone`.
- `--result-policy-anchor-weight 0.01`.
- `--result-policy-anchor-mode kl`.
- LR `3e-4`, 400 steps, snapshots every 50.

Specific cells:

| Cell | Run directory |
| --- | --- |
| `src4_add2` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_tiny_anchor/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_freeze_policy_backbone_anchor0.01_steps400/2026-05-28_191912_530071_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_tiny_anchor/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_freeze_policy_backbone_anchor0.01_steps400/2026-05-28_191912_530116_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Final eval | Best normal | Last normal | Last inj-zero | Last forced-random | Last oracle | Last calc | Anchor agree | Anchor acc | Anchor loss | Trainable groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src4_backbone_anchor0.01` | `0.7125` | `0.7250` at `400` | `0.7250` | `0.0325` | `0.0950` | `0.7625` | `0.8200` | `1.0000` | `0.8450` | `0.000004` | `calculator_hook.result_proj`, `upstream` |
| `src5_backbone_anchor0.01` | `0.8600` | `0.8700` at `50` | `0.8650` | `0.0075` | `0.0400` | `0.8725` | `0.8000` | `0.9975` | `0.8225` | `0.000004` | `calculator_hook.result_proj`, `upstream` |

## Conclusion

Tiny anchoring is redundant under policy-backbone freezing. The policy barely
drifted, and final answer accuracy was slightly worse than the no-anchor
policy-backbone-freeze result.

Label:

```text
bottleneck_to_additive_policy_backbone_tiny_anchor_no_gain
```

## Anti-Rerun Note

Do not repeat `--freeze-calculator-policy-backbone` plus fixed KL anchor
`0.01`, LR `3e-4`, 400-step unfreeze from the same adapted
`src4_add2/src5_add5` checkpoints as novelty.
