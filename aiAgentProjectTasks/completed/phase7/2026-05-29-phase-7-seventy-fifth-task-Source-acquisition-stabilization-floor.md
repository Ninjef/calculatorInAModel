# Phase 7 Seventy-Fifth Task: Source Acquisition Stabilization Floor

## Status

Completed 2026-05-29.

## Question

Does keeping entropy/diversity/improvement-assignment source stabilization
active prevent the decay-to-zero collapse, and does the resulting source
transfer into additive non-bottleneck handoff?

## Setup

- Trained a fresh bottleneck source with the current direct-feedback
  result-space recipe.
- Added `result_policy_entropy_weight=0.05` and
  `result_policy_batch_diversity_weight=0.1`.
- Kept `result_policy_improvement_assignment_weight=10` active with no decay.
- Used exact-grid natural `0..19`, frozen product semantic decoder, CLI seed
  `9`, and 1600 source steps.
- Ran 800-step frozen-policy additive handoffs for source step `1400` and
  final weights.

## Result

| Candidate | Source normal | Normal @ 600 | Final handoff |
| --- | ---: | ---: | ---: |
| step `1400` | `0.9100` | `0.4575` | `0.4425` |
| final | `0.8575` | `0.5250` | `0.6500` |

The source did not collapse: it reached `0.9100` at step `1400` and final eval
`0.8575`. However, additive handoff remained weak. Final transferred better
than the higher-source-normal step `1400`, but only reached `0.6500`.

## Decision

```text
source_acquisition_entropy_diversity_nodecay_source_positive_transfer_negative
```

Persistent source stabilization fixes the collapse but does not produce
handoff-friendly source geometry. Continuation/readout was skipped because the
best handoff was below the recent weak-source boundary.

## Next

Optimize source acquisition for 600-step handoff/continuation slope, or add a
handoff-aware proxy/anchor rather than only improving bottleneck source
accuracy.
