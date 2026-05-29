# Phase 7 Fifty-Fourth Task: Policy-Backbone Long Adaptation

## Status

Completed 2026-05-29.

## Question

With the calculator policy backbone frozen and calculator use preserved, can
longer downstream/readout adaptation close the weak-source handoff gap?

## Setup

- Continued from the adapted weak-source frozen-policy handoff checkpoints.
- Loaded full model checkpoints.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 1600 steps with snapshots every 100 steps.

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` policy-backbone long | `0.7550` | `0.7600` at `1100` | `0.8550` | `0.0525` | `0.0900` |
| `src5_add5` policy-backbone long | `0.9500` | `0.9625` at `1500` | `0.8325` | `0.0025` | `0.0350` |

## Decision

```text
bottleneck_to_additive_policy_backbone_long_adaptation_mixed
```

Longer stable-policy adaptation makes `src5_add5` strong without an anchor,
but it does not rescue `src4_add2`. More readout time alone does not erase
source sensitivity.

## Next

Focus on source-policy acquisition/selection, stronger readout objectives for
weak sources, or utility-aware stable-policy adaptation.
