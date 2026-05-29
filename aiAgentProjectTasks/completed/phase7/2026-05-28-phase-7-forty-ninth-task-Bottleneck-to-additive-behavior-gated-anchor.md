# Phase 7 Forty-Ninth Task: Bottleneck-to-Additive Behavior-Gated Anchor

## Status

Completed on 2026-05-28.

## Question

Can behavior-gated anchoring preserve the transferred non-bottleneck calculator
policy with a lower average anchor weight than a fixed `0.1` anchor?

## Code Change

Added behavior-gated result-policy anchoring to `scripts/overfit_one_batch.py`:

- `--result-policy-anchor-gate-threshold`
- `--result-policy-anchor-gate-weight`
- `--result-policy-anchor-gate-metric`

The training curve now logs base/effective anchor weight, gate metric/value,
and whether the gate was active.

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Use base `--result-policy-anchor-weight 0.01`.
- Use `--result-policy-anchor-mode kl`.
- Use `--result-policy-anchor-gate-threshold 0.9`.
- Use `--result-policy-anchor-gate-weight 0.1`.
- Use `--result-policy-anchor-gate-metric argmax_agreement`.
- Train for `400` steps.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_gated_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_gate0.9_to0.1_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_gated_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_gate0.9_to0.1_steps400
```

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` gated | `0.8050` | `0.8025` at `400` | `0.7700` | `0.0400` | `4/9` | `0.0500` |
| `src5_add5` gated | `0.9675` | `0.9525` at `350` | `0.7700` | `0.0000` | `8/9` | `0.0900` |

## Conclusion

Label:

```text
bottleneck_to_additive_behavior_gated_anchor_partial
```

The gate worked mechanically and improved over constant anchor `0.01`, but it
did not beat constant anchor `0.1`. It is useful infrastructure for adaptive
retention, not yet evidence that discrete threshold gating is the better
handoff recipe.

Do not repeat this exact gate as novelty. Next adaptive-retention work should
change the metric, thresholding shape, or couple the gate to calculator-result
accuracy.
