# Phase 7 Fifty-Third Task: Policy-Backbone Freeze Plus Tiny Anchor

## Status

Completed 2026-05-29.

## Question

Does a tiny result-policy anchor help when the calculator policy backbone is
already frozen and only the result action head plus downstream/readout path can
adapt?

## Setup

- Continued from the adapted weak-source frozen-policy handoff checkpoints.
- Loaded full model checkpoints.
- Used `--freeze-calculator-policy-backbone`.
- Used fixed KL result-policy anchor `0.01`.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 400 steps with snapshots every 50 steps.

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Final anchor agreement | Anchor loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` backbone + anchor `0.01` | `0.7125` | `0.7250` at `400` | `0.8200` | `0.0325` | `1.0000` | `0.000004` |
| `src5_add5` backbone + anchor `0.01` | `0.8600` | `0.8700` at `50` | `0.8000` | `0.0075` | `0.9975` | `0.000004` |

## Decision

```text
bottleneck_to_additive_policy_backbone_tiny_anchor_no_gain
```

The tiny anchor did not improve over no-anchor policy-backbone freezing. The
policy was already stable, so the branch needs better downstream/readout
adaptation or source-policy acquisition rather than more tiny action-policy
retention.

## Next

Do not repeat this exact tiny-anchor combination as novelty. Consider
answer-utility-aware retention or stronger downstream adaptation under a stable
policy.
