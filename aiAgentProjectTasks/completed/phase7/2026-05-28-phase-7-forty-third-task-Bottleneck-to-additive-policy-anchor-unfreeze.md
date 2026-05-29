# Phase 7 Forty-Third Task: Bottleneck-to-Additive Policy-Anchor Unfreeze

## Status

Completed on 2026-05-28.

## Question

Can an explicit result-policy anchor prevent the calculator-policy collapse
seen under plain low-LR full unfreeze while still allowing useful
non-bottleneck adaptation?

## Implementation

- Added `--result-policy-anchor-weight`.
- Added `--result-policy-anchor-decay-steps`.
- Added `--result-policy-anchor-temperature`.
- Added `--result-policy-anchor-mode {kl,mse}`.

The anchor snapshots the initial fixed-grid result-space policy and penalizes
KL or logit-MSE drift. It currently requires `--exhaustive-grid-batch`.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_unfreeze
```

Configuration:

- resumed from adapted weak-source additive checkpoints;
- `--semantic-decoder-checkpoint-load-scope full_model`;
- no `--freeze-calculator-policy`;
- global LR `3e-4`;
- `--result-policy-anchor-weight 10`;
- `--result-policy-anchor-mode kl`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

## Result

| Run | Frozen adapted final | Plain unfreeze final | Anchored final | Last injection-zero | Last oracle | Last learned calc | Anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor | `0.6050` | `0.5200` | `0.7475` | `0.0100` | `0.7875` | `0.8075` | `0.9800` |
| `src5_add5` anchor | `0.8175` | `0.8100` | `0.9525` | `0.0000` | `0.9375` | `0.7950` | `0.9850` |

## Decision

```text
bottleneck_to_additive_policy_anchor_unfreeze_partial
```

Anchored unfreeze prevents the policy collapse seen in the low-LR unfreeze
negative and improves both adapted weak-source handoffs. This is still staged
and anchored, not final scalable from-scratch discovery.

## Next

- Test anchor decay/off-ramp schedules.
- Test selective unfreezing with policy retention.
- Look for less prescriptive ways to acquire the source policy.
