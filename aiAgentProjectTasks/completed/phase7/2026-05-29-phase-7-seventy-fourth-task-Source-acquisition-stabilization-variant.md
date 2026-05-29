# Phase 7 Seventy-Fourth Task: Source Acquisition Stabilization Variant

## Status

Completed 2026-05-29.

## Question

Can small entropy plus batch-diversity regularization stabilize fresh
bottleneck source acquisition enough to improve the weak-source boundary before
downstream non-bottleneck handoff?

## Setup

- Trained a fresh bottleneck source with the current direct-feedback
  result-space recipe.
- Added `result_policy_entropy_weight=0.05` and
  `result_policy_batch_diversity_weight=0.1`.
- Decayed all result-policy stabilization terms to zero over 1200 steps.
- Used exact-grid natural `0..19`, frozen product semantic decoder, CLI seed
  `9`, and 1600 source steps.

## Result

| Step | Source normal | Injection-zero | Learned calc |
| --- | ---: | ---: | ---: |
| `700` | `0.7050` | `0.0650` | `0.7050` |
| `900` | `0.7050` | `0.0525` | `0.7050` |
| `1200` | `0.5800` | `0.0400` | `0.5800` |
| `1300` | `0.1900` | `0.0575` | `0.1900` |
| `1600` | `0.2175` | `0.0300` | `0.2175` |
| final eval | `0.1825` | `0.0703` | `0.2266` |

## Decision

```text
source_acquisition_entropy_diversity_decay_negative
```

The variant peaked before the decay completed, then collapsed after the active
source objective reached zero. No downstream handoff was run.

## Next

Avoid pure decay-to-zero source stabilization. If entropy/diversity is reused,
keep a nonzero floor, add anchoring, or optimize source acquisition against a
handoff/continuation proxy.
