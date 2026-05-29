# Phase 7 Seventy-Eighth Task: Stabilized Source Continuation Boundary

## Status

Completed 2026-05-29.

## Question

How far can the no-decay stabilized source continuation budget be reduced
before the 600-step readout falls below the non-bottleneck gate?

## Setup

- Used the prior frozen-policy continuation run from the no-decay final-source
  handoff.
- Loaded step `600`, `500`, `400`, and `300` continuation checkpoints.
- Ran 600 policy-backbone-frozen readout steps from each checkpoint.

## Result

| Continuation checkpoint | Readout final eval | Best snapshot | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: |
| step `600` | `0.9425` | `0.9425` | `0.0078` | `0.0781` |
| step `500` | `0.9400` | `0.9400` | `0.0156` | `0.0703` |
| step `400` | `0.9175` | `0.9325` | `0.0078` | `0.0703` |
| step `300` | `0.8850` | `0.9150` | `0.0078` | `0.0625` |

## Decision

```text
stabilized_source_400_continuation_boundary_positive_300_negative
```

For this lineage, 400 continuation steps are enough to pass by final eval, but
300 continuation steps are not.

## Next

Replicate the 400/500 continuation boundary on another no-decay stabilized
source before changing the default continuation budget.
