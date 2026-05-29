# Phase 7 Seventy-Seventh Task: Stabilized Source Reduced Continuation

## Status

Completed 2026-05-29.

## Question

Can the no-decay stabilized source lineage still clear the non-bottleneck gate
if continuation is reduced from 800 to 600 steps?

## Setup

- Started from the 600-step checkpoint of the prior frozen-policy continuation
  run.
- Ran the standard 600-step policy-backbone-frozen readout.
- Compared against the prior 800-continuation plus 600-readout positive.

## Result

| Recipe | Final eval | Best snapshot | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: |
| 800 continuation + 600 readout | `0.9575` | `0.9625` | `0.0156` | `0.0859` |
| 600 continuation + 600 readout | `0.9425` | `0.9425` | `0.0078` | `0.0781` |

## Decision

```text
stabilized_source_600_continuation_readout_positive
```

The reduced continuation still clears the `0.90` non-bottleneck gate with
calculator-dependence controls far below normal.

## Next

Replicate this reduced continuation on another no-decay stabilized source
before making 600 continuation the default.
