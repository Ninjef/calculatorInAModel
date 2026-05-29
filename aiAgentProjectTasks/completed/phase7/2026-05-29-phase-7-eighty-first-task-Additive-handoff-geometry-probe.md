# Phase 7 Eighty-First Task: Additive Handoff Geometry Probe

## Status

Completed 2026-05-29.

## Question

Can a direct injection-to-answer or short-slope diagnostic replace the
expensive additive handoff probe as a source-transfer selector?

## Setup

- Added `scripts/run_additive_handoff_geometry_probe.py`.
- Loaded source checkpoints in additive-compatible mode.
- Froze the calculator policy.
- Measured forced-result counterfactual losses over all result classes.
- Measured short downstream-only learning slope at steps `0,50,100`.
- Tested seed-9 positive, seed-10 weak/negative checkpoints, `src6` positive,
  and `src7` boundary-negative.

## Result

| Source | Known status | Calc | True-best | True top-3 | True-best gap | 100-step loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seed-9 final | positive | `0.8675` | `0.0625` | `0.2125` | `0.0034` | `1.5852` |
| seed-10 step `1000` | weak | `0.7550` | `0.0000` | `0.0300` | `0.0061` | `1.6569` |
| seed-10 step `1300` | weak | `0.8575` | `0.0000` | `0.0300` | `0.0063` | `1.5986` |
| seed-10 step `1400` | weak | `0.8475` | `0.0000` | `0.0300` | `0.0058` | `1.5955` |
| seed-10 final | negative | `0.9025` | `0.0000` | `0.0450` | `0.0063` | `1.6221` |
| `src6` final | positive | `0.8450` | `0.0050` | `0.0550` | `0.0049` | `1.7164` |
| `src7` step `1400` | boundary negative | `0.8000` | `0.0050` | `0.0950` | `0.0048` | `1.8318` |

## Decision

```text
additive_handoff_geometry_probe_partial_no_selector
```

The forced-result geometry metric is useful as a warning signal for the
seed-10 hostile geometry, but it is not a validated selector and cannot replace
the 400/600-step additive handoff probe.

## Next

Use this metric as a source-training diagnostic or auxiliary target candidate,
while keeping actual additive handoff probes as the selection gate.
