# Phase 7 Seventy-Third Task: `src7` 600-Step Selector Replication

## Status

Completed 2026-05-29.

## Question

Does 600-step handoff selection and the reduced selected-source recipe
replicate on another fresh source family?

## Setup

- Trained fresh bottleneck `src7` with checkpoint snapshots.
- Probed step `1000`, step `1400`, and final with 800-step additive
  frozen-policy handoffs.
- Continued the 600-step-selected winner, step `1400`, for 800 more
  frozen-policy steps.
- Ran 600 policy-backbone-frozen readout steps from the continued checkpoint.

## Result

| Candidate | Source normal | Normal @ 600 | Final handoff |
| --- | ---: | ---: | ---: |
| step `1000` | `0.7075` | `0.4850` | `0.5375` |
| step `1400` | `0.7500` | `0.5025` | `0.7325` |
| final | `0.8100` | `0.4150` | `0.5000` |

| Stage | Final eval | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: |
| selected handoff | `0.7325` | `0.0234` | `0.0391` |
| 800-step continuation | `0.8125` | `0.0469` | `0.0391` |
| 600-step readout | `0.8825` | `0.0625` | `0.0703` |

## Decision

```text
src7_600_step_selector_positive_recipe_boundary_negative
```

The 600-step selector picked the best full-handoff candidate, but the selected
source family was too weak for the reduced continuation/readout recipe to clear
the `0.90` gate.

## Next

Focus on source acquisition optimized for 600-step handoff/continuation slope,
not more source-normal-accuracy selection.
