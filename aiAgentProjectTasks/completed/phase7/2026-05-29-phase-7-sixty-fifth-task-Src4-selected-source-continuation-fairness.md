# Phase 7 Sixty-Fifth Task: `src4` Selected-Source Continuation Fairness

## Status

Completed 2026-05-29.

## Question

Does the fair continuation recipe that nearly closed the `src5` selected-source
gap also improve the weaker `src4` step-1200 selected-source lineage?

## Setup

- Started from the `src4` step-1200 800-step frozen-policy additive handoff.
- Ran another 800-step frozen-policy continuation with LR `3e-3`.
- Then ran 1600-step no-anchor stable-policy adaptation with
  `--freeze-calculator-policy-backbone` and LR `3e-4`.
- Used additive, non-bottleneck result-space calculator mode throughout.
- Used exact-grid natural `0..19`, answer loss weight `1`, and frozen semantic
  decoder.

## Result

| Run | Start final | Continued final | Long final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `src4` final-source lineage | n/a | `0.6050` | `0.7550` | `0.7600` at `1100` | `0.8550` | `0.0525` | `0.0900` |
| selected `src4` direct long | `0.7800` | n/a | `0.8900` | `0.8975` at `1400` | `0.8225` | `0.0000` | `0.0175` |
| selected `src4` continued long | `0.7800` | `0.8150` | `0.9125` | `0.9475` at `1400` | `0.8025` | `0.0075` | `0.0250` |

## Decision

```text
src4_selected_source_continuation_fairness_positive
```

The fair continuation recipe improves `src4` as well: it beats the direct
selected-source long adaptation and the old final-source long adaptation, while
remaining calculator-dependent under controls.

## Next

Reduce the cost of the handoff-probe/continuation recipe, or train source
policies to optimize early handoff and continuation slope directly.
