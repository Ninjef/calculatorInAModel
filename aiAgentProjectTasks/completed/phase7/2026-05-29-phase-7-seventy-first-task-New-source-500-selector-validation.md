# Phase 7 Seventy-First Task: New-Source 500-Step Selector Validation

## Status

Completed 2026-05-29.

## Question

Does the shortened 500-step handoff selector generalize to a newly acquired
source family?

## Setup

- Trained fresh bottleneck `src6` with checkpoint snapshots every 100 steps.
- Probed source checkpoints step `1200`, step `1500`, and final with additive
  seed `6`.
- Ran each additive handoff for 800 steps so the 500-step selector score could
  be checked against the full handoff result.

## Result

| Candidate | Source normal | Normal @ 500 | Normal @ 600 | Final eval |
| --- | ---: | ---: | ---: | ---: |
| step `1200` | `0.8350` | `0.6025` | `0.7075` | `0.8450` |
| step `1500` | `0.8625` | `0.7200` | `0.7800` | `0.8875` |
| final | `0.8850` | `0.6850` | `0.8050` | `0.8975` |

## Decision

```text
new_source_500_step_selector_generalization_negative
```

The 500-step selector would pick step `1500`, but the full 800-step handoff was
best from the final source checkpoint. The 600-step selector would have picked
the full-handoff winner.

## Next

For newly acquired source families, use the 600-step selector or full
confirmation until a cheaper proxy is validated. The fresh `src6` final handoff
is near-gate (`0.8975`), so the next non-duplicative test is selected-source
continuation/readout from that checkpoint or acquisition tuned for 600-step
handoff slope.
