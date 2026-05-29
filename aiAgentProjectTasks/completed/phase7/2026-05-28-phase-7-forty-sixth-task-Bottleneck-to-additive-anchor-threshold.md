# Phase 7 Forty-Sixth Task: Bottleneck-to-Additive Anchor Threshold

## Status

Completed on 2026-05-28.

## Question

Is constant KL result-policy anchor weight `0.01` enough to preserve
calculator use during non-bottleneck full-policy unfreezing?

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Use `--result-policy-anchor-mode kl`.
- Use constant `--result-policy-anchor-weight 0.01`.
- Train for `400` steps.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_threshold_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_threshold_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_steps400
```

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor `0.01` | `0.7850` | `0.7850` at `400` | `0.7625` | `0.0050` | `0.8825` |
| `src5_add5` anchor `0.01` | `0.9375` | `0.9250` at `400` | `0.6425` | `0.0000` | `0.7050` |

## Conclusion

Label:

```text
bottleneck_to_additive_anchor_0p01_threshold_mixed
```

Anchor `0.01` is enough to keep the model using the calculator path, but it is
not enough for clean policy retention across the two-cell gate. It is weaker
than anchor `0.1` on both answer accuracy and calculator-result preservation,
with a sharp policy-retention drop in `src5_add5`.

Do not repeat this exact anchor `0.01` gate as novelty. Future schedules should
test a floor or gate around the `0.1` region, selective unfreezing, or source
policies that need less active retention.
