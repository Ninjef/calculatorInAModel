# Phase 7 Eighty-Second Task: Additive Geometry Selector Validation

## Status

Completed 2026-05-29.

## Question

Can the additive handoff geometry probe select known source-checkpoint winners
inside existing `src2`, `src4`, and `src5` source families?

## Setup

- Ran `scripts/run_additive_handoff_geometry_probe.py` with `--slope-steps 0`.
- Probed `src5` step `1100/1400/1500/final`.
- Probed `src4` step `1000/1200/final`.
- Probed `src2` step `1300/final`.
- Compared geometry metrics to already-known handoff outcomes.

## Result

| Source | Known handoff result | Winner? | True-best | True top-3 | True-best gap |
| --- | ---: | --- | ---: | ---: | ---: |
| `src5` step `1100` | `0.7950` full | yes | `0.0325` | `0.1375` | `0.0038` |
| `src5` step `1400` | `0.6400` full | no | `0.0075` | `0.1350` | `0.0037` |
| `src5` step `1500` | `0.6975` full | no | `0.0075` | `0.1325` | `0.0035` |
| `src5` final | `0.5550` full | no | `0.0125` | `0.1350` | `0.0033` |
| `src4` step `1000` | `0.5225` full | no | `0.0600` | `0.1350` | `0.0036` |
| `src4` step `1200` | `0.7800` full | yes | `0.0600` | `0.1325` | `0.0035` |
| `src4` final | no full comparison here | no | `0.0600` | `0.1100` | `0.0030` |
| `src2` step `1300` | `0.8675` full | no | `0.0325` | `0.0350` | `0.0042` |
| `src2` final | `0.9525` full | yes | `0.0325` | `0.0350` | `0.0039` |

## Decision

```text
additive_geometry_selector_validation_negative
```

The geometry probe is useful as a warning diagnostic, but it cannot replace
actual additive handoff probes as a source-checkpoint selector.

## Next

Keep 400/600-step handoff probes as selection gates. Use geometry as logging
or pair it with a stronger one/few-update slope proxy before making it a source
objective.
