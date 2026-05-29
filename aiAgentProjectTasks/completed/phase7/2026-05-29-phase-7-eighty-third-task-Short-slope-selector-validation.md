# Phase 7 Eighty-Third Task: Short Slope Selector Validation

## Status

Completed 2026-05-29.

## Question

Can a 25/50/100-step downstream loss-slope proxy replace the 400/500/600-step
additive handoff selector for choosing source checkpoints?

## Setup

- Ran `scripts/run_additive_handoff_geometry_probe.py` with
  `--slope-steps 0,25,50,100`.
- Probed `src5` step `1100/1400/1500/final`.
- Probed `src4` step `1000/1200/final`.
- Probed `src2` step `1300/final`.
- Compared 100-step loss and loss drop to already-known handoff winners.

## Result

| Source | Known handoff result | Winner? | 100-step loss | 100-step loss drop |
| --- | ---: | --- | ---: | ---: |
| `src5` step `1100` | `0.7950` full | yes | `1.7956` | `0.8238` |
| `src5` step `1400` | `0.6400` full | no | `1.7752` | `0.8464` |
| `src5` step `1500` | `0.6975` full | no | `1.7686` | `0.8535` |
| `src5` final | `0.5550` full | no | `1.7684` | `0.8549` |
| `src4` step `1000` | `0.5225` full | no | `1.7128` | `0.8984` |
| `src4` step `1200` | `0.7800` full | yes | `1.8478` | `0.7672` |
| `src4` final | no full comparison here | no | `1.8721` | `0.7493` |
| `src2` step `1300` | `0.8675` full | no | `1.6639` | `0.9513` |
| `src2` final | `0.9525` full | yes | `1.6424` | `0.9778` |

## Decision

```text
short_slope_selector_validation_negative
```

100-step loss slope selects the wrong checkpoint for `src5` and `src4`; it
cannot replace the established handoff selector.

## Next

Keep 500/600-step handoff probes as the selection gate. Use shorter slope only
as exploratory logging unless a learned proxy is validated against the existing
handoff trace corpus.
