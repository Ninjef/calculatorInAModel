# Phase 7 Forty-Fourth Task: Bottleneck-to-Additive Anchor Decay Off-Ramp

## Status

Completed on 2026-05-28.

## Question

Can the explicit KL result-policy anchor from the controlled-unfreeze partial
positive be removed during training without losing calculator use?

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Use `--result-policy-anchor-weight 10`.
- Use `--result-policy-anchor-decay-steps 200`.
- Train for `400` steps so the final half runs with anchor weight `0`.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_decay_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor10_decay200_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_decay_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor10_decay200_steps400
```

## Result

| Run | Step-200 calc | Final calc | Final eval | Best normal | Final injection-zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` decay200 | `0.8300` | `0.5950` | `0.5925` | `0.7250` at `250` | `0.0325` |
| `src5_add5` decay200 | `0.8225` | `0.3850` | `0.6750` | `0.9575` at `200` | `0.0375` |

## Conclusion

Label:

```text
bottleneck_to_additive_anchor_decay_offramp_negative
```

The policies were still usable when the anchor reached zero, but they drifted
during the no-anchor tail. The constant anchor was not just an optimization
aid; it was actively preserving the non-bottleneck calculator policy.

Do not repeat this same fast `200/400` decay schedule as novelty. Next
anchored-unfreeze work should use slower/floored/gated anchors, selective
unfreezing, or source-policy acquisition that is robust without anchoring.
