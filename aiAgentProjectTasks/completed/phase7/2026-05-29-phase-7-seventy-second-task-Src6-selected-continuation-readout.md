# Phase 7 Seventy-Second Task: `src6` Selected Continuation and Readout

## Status

Completed 2026-05-29.

## Question

Does the selected-source continuation/readout recipe rescue the fresh `src6`
near-gate final-source handoff?

## Setup

- Started from fresh `src6` final-source 800-step frozen-policy handoff
  (`0.8975` final eval).
- Ran 800 more frozen-policy continuation steps.
- Ran 600 policy-backbone-frozen readout steps from the continued checkpoint.

## Result

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial handoff | `0.8975` | `0.8975` at step `800` | `0.0469` | `0.0703` | `0.9063` | `0.8594` |
| 800-step continuation | `0.9625` | `0.9675` at step `700` | `0.0234` | `0.0703` | `0.9375` | `0.8594` |
| 600-step readout | `0.9850` | `0.9900` at step `500` | `0.0625` | `0.0547` | `0.9297` | `0.8594` |

## Decision

```text
src6_selected_continuation_readout_positive
```

The reduced selected-source recipe clears the non-bottleneck gate on a fresh
source family. Continuation alone crosses `0.90`; 600-step readout improves the
final answer score further while controls remain far below normal.

## Next

Validate another fresh source with the 600-step selector and reduced recipe, or
shift source acquisition toward 600-step handoff/continuation slope.
