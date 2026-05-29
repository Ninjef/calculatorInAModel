# 2026-05-29 Policy-Backbone Long Adaptation

## Aim

Test whether the policy-backbone-freeze branch is limited mainly by training
time for the downstream/additive readout.

## Runs

Run root:

```text
runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_long_adaptation
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- Continued from adapted weak-source frozen-policy handoff checkpoints.
- Full model checkpoint load.
- `--freeze-calculator-policy-backbone`.
- No result-policy anchor.
- LR `3e-4`, 1600 steps, snapshots every 100.

Specific cells:

| Cell | Run directory |
| --- | --- |
| `src4_add2` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_long_adaptation/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_freeze_policy_backbone_steps1600/2026-05-28_192420_853459_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_backbone_freeze_long_adaptation/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_freeze_policy_backbone_steps1600/2026-05-28_192420_853508_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Final eval | Best normal | Last normal | Last inj-zero | Last forced-random | Last oracle | Last calc | Trainable groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `src4_backbone_long1600` | `0.7550` | `0.7600` at `1100` | `0.7175` | `0.0525` | `0.0900` | `0.7325` | `0.8550` | `calculator_hook.result_proj`, `upstream` |
| `src5_backbone_long1600` | `0.9500` | `0.9625` at `1500` | `0.9550` | `0.0025` | `0.0350` | `0.9050` | `0.8325` | `calculator_hook.result_proj`, `upstream` |

## Conclusion

Long stable-policy adaptation is enough for `src5_add5` to become a strong
non-bottleneck calculator-dependent handoff without an anchor. It does not
rescue `src4_add2`, even though the learned calculator-result accuracy remains
high. Source handoff quality is still the bottleneck.

Label:

```text
bottleneck_to_additive_policy_backbone_long_adaptation_mixed
```

## Anti-Rerun Note

Do not repeat no-anchor `--freeze-calculator-policy-backbone`, LR `3e-4`,
1600-step adaptation from the same adapted `src4_add2/src5_add5` checkpoints
as novelty.
