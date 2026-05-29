# Phase 7 Forty-Seventh Task: Bottleneck-to-Additive Anchor Floor Schedule

## Status

Completed on 2026-05-28.

## Question

Can the result-policy anchor decay from a stronger initial weight to a
lightweight nonzero floor, preserving calculator use without the failed
zero-anchor tail?

## Code Change

Added `--result-policy-anchor-floor` to `scripts/overfit_one_batch.py`.
The result-policy anchor can now linearly decay to a configured floor instead
of always decaying to zero.

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Use `--result-policy-anchor-mode kl`.
- Use `--result-policy-anchor-weight 1`.
- Use `--result-policy-anchor-decay-steps 200`.
- Use `--result-policy-anchor-floor 0.1`.
- Train for `400` steps.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_floor_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor1_decay200_floor0.1_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_floor_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor1_decay200_floor0.1_steps400
```

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Final anchor weight | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` floor `0.1` | `0.7925` | `0.7725` at `200` | `0.8175` | `0.0250` | `0.1000` | `0.9225` |
| `src5_add5` floor `0.1` | `0.9775` | `0.9650` at `350` | `0.7800` | `0.0075` | `0.1000` | `0.8975` |

## Conclusion

Label:

```text
bottleneck_to_additive_anchor_floor_schedule_partial
```

A nonzero floor rescues the failed zero-off-ramp shape. It preserves
calculator dependence and useful policy accuracy, but it does not outperform
constant anchor `0.1` in this gate. The mechanism is useful for future
retention schedules, especially if the floor can be made adaptive or gated by
calculator-result accuracy.
