# Phase 7 Fifty-Second Task: Bottleneck-to-Additive Policy-Backbone Freeze

## Status

Completed 2026-05-29.

## Question

Can we prevent no-anchor policy collapse by freezing only the pre-hook
calculator-policy representation while leaving the action head trainable?

## Setup

- Added `--freeze-calculator-policy-backbone`.
- Continued from the adapted weak-source frozen-policy handoff checkpoints.
- Loaded full model checkpoints and unfroze the full policy except the
  policy backbone.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Used no result-policy anchor.
- Ran 400 steps with snapshots every 50 steps.

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Trainable groups |
| --- | ---: | ---: | ---: | ---: | --- |
| `src4_add2` policy-backbone freeze | `0.7250` | `0.7250` at `400` | `0.8200` | `0.0400` | `calculator_hook.result_proj`, `upstream` |
| `src5_add5` policy-backbone freeze | `0.8700` | `0.8750` at `300` | `0.8025` | `0.0075` | `calculator_hook.result_proj`, `upstream` |

## Decision

```text
bottleneck_to_additive_policy_backbone_freeze_partial
```

Freezing the policy backbone avoids the learned-calculator collapse seen in
plain no-anchor unfreeze and action-head-only freezing. It improves over the
frozen-adapted weak baselines, but it remains below lightweight anchored
unfreezing.

## Next

Combine stable policy-backbone freezing with a lightweight/utility-aware
retention signal, improve source-policy acquisition, or test a different
selective parameter set.
