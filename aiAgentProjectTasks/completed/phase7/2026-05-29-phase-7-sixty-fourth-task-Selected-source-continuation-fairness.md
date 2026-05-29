# Phase 7 Sixty-Fourth Task: Selected-Source Continuation Fairness

## Status

Completed 2026-05-29.

## Question

Was the old `src5` final-source long-adaptation advantage caused by the
checkpoint itself, or because that lineage received an extra 800-step
frozen-policy continuation before the 1600-step stable-policy adaptation?

## Setup

- Started from the `src5` step-1100 800-step frozen-policy additive handoff.
- Ran another 800-step frozen-policy continuation with LR `3e-3`.
- Then ran 1600-step no-anchor stable-policy adaptation with
  `--freeze-calculator-policy-backbone` and LR `3e-4`.
- Used additive, non-bottleneck result-space calculator mode throughout.
- Used exact-grid natural `0..19`, answer loss weight `1`, and frozen semantic
  decoder.

## Result

| Run | Start final | Continued final | Long final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `src5` final-source lineage | `0.5550` | `0.8175` | `0.9500` | `0.9625` at `1500` | `0.8325` | `0.0025` | `0.0350` |
| selected `src5` step-1100 lineage | `0.7950` | `0.8800` | `0.9425` | `0.9625` at `800` | `0.8275` | `0.0000` | `0.0225` |

## Decision

```text
selected_source_continuation_fairness_positive
```

The old final-source advantage was mostly a continuation-depth fairness issue.
With the same extra frozen-policy continuation, the handoff-probe-selected
step-1100 lineage nearly matched the old final-source long-adaptation result.

## Next

Apply the same fair continuation recipe to `src4` step-1200, or optimize source
acquisition for early handoff and continuation slope rather than source action
accuracy.
