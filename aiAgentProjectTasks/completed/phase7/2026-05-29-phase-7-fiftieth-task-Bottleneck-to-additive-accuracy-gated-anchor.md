# Phase 7 Fiftieth Task: Bottleneck-to-Additive Accuracy-Gated Anchor

## Status

Completed 2026-05-29.

## Question

Can the result-policy anchor stay at a low base weight and boost only when the
current learned calculator-result accuracy falls below threshold?

## Setup

- Continued from the adapted weak-source frozen-policy handoff checkpoints.
- Loaded full model checkpoints and unfroze the policy.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Used base KL result-policy anchor `0.01`.
- Used gate metric `current_argmax_accuracy`.
- Tested gate thresholds `0.80` and `0.82`, gate weight `0.1`.
- Ran 400 steps with snapshots every 50 steps.

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` gate `0.80` | `0.7725` | `0.7975` at `400` | `0.8100` | `0.0500` | `0/9` | `0.0100` |
| `src5_add5` gate `0.80` | `0.9825` | `0.9625` at `400` | `0.7900` | `0.0000` | `8/9` | `0.0900` |
| `src4_add2` gate `0.82` | `0.7900` | `0.8075` at `150` | `0.8000` | `0.0775` | `4/9` | `0.0500` |
| `src5_add5` gate `0.82` | `0.9825` | `0.9675` at `400` | `0.7900` | `0.0000` | `8/9` | `0.0900` |

## Decision

```text
bottleneck_to_additive_accuracy_gated_anchor_mixed_no_go
```

The gate behaved adaptively and helped `src5_add5`, but it did not beat fixed
anchor `0.1` across both weak-source cells because `src4_add2` remained below
the fixed-anchor answer result. Do not repeat these exact thresholds as
novelty.

## Next

Try continuous/adaptive anchor control, selective policy-path unfreezing, a
retention signal that combines calculator accuracy with answer utility, or
better source-policy acquisition.
