# 2026-05-29 Probe-Selected Policy-Backbone Adaptation

## Aim

Test whether source checkpoints selected by the 600-step additive handoff probe
also reduce the need for later anchoring or long downstream adaptation.

## Runs

Run root:

```text
runs/2026-05-29_phase7_probe_selected_policy_backbone_adaptation
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- `calculator_injection_mode=add`.
- Continued from the probe-selected frozen-policy additive handoff checkpoints.
- Loaded full model checkpoints.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- LR `3e-4`, answer loss weight `1`, 1600 steps, snapshots every 100.

Specific cells:

| Cell | Run directory |
| --- | --- |
| `src4_step1200_add2` | `runs/2026-05-29_phase7_probe_selected_policy_backbone_adaptation/source_seed4_step1200_additive_seed2_selected_freeze_policy_backbone_steps1600/2026-05-28_203408_536751_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_step1100_add5` | `runs/2026-05-29_phase7_probe_selected_policy_backbone_adaptation/source_seed5_step1100_additive_seed5_selected_freeze_policy_backbone_steps1600/2026-05-28_203408_536697_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Frozen handoff final | Adapted final eval | Adapted best normal | Last inj-zero | Last forced-random | Last oracle | Last calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_step1200_add2_selected` | `0.7800` | `0.8900` | `0.8975` at `1400` | `0.0000` | `0.0175` | `0.8675` | `0.8225` |
| `src5_step1100_add5_selected` | `0.7950` | `0.9250` | `0.9325` at `1600` | `0.0000` | `0.0150` | `0.9525` | `0.8275` |

Comparison to older final-source long adaptation:

| Seed pair | Selected-source adapted final | Old final-source adapted final | Difference |
| --- | ---: | ---: | ---: |
| `src4/add2` | `0.8900` | `0.7550` | `+0.1350` |
| `src5/add5` | `0.9250` | `0.9500` | `-0.0250` |

## Conclusion

Probe-selected sources can make later stable-policy adaptation much better for
weak handoffs: `src4` improved from the old final-source long-adaptation result
of `0.7550` to `0.8900`, while remaining calculator-dependent under controls.

The effect is not universal. For `src5`, the probe-selected checkpoint improved
early frozen handoff, but the older final-source checkpoint still produced a
better 1600-step stable-policy adaptation result.

Label:

```text
probe_selected_policy_backbone_adaptation_mixed_positive
```

## Anti-Rerun Note

Do not repeat no-anchor `--freeze-calculator-policy-backbone`, LR `3e-4`,
1600-step adaptation from the same probe-selected `src4` step-1200 or `src5`
step-1100 additive handoff checkpoints as novelty.

Next useful tests should ask whether the source selector needs a second-stage
long-adaptation/readout-compatibility score, or whether source acquisition can
optimize both the 600-step frozen handoff slope and later readout adaptability.
