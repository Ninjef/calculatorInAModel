# Phase 7 Thirtieth Task: Optimizer Step Trust Region Gate

## Purpose

Test whether bounding actual optimizer parameter movement rescues refreshed
online-shadow Stage 1 training.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Calibrated module: h32 validation-gradient module.
- Shadow weight: `1.0`.
- Apply norm clamp: `10`.
- Refresh cadence: every `50` training steps.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Runs

| Max delta | Final exact | Best snapshot | Min/median/last trust scale | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.075` | `0.060` | `0.251 / 0.300 / 0.259` | `0.0475` | `5.70` |
| `0.10` | `0.040` | `0.045` | `0.550 / 0.624 / 0.594` | `0.0325` | `10.00` |

## Conclusion

```text
optimizer_step_trust_region_stage1_negative
```

Bounding the realized AdamW update stabilized shadow-feedback norms and kept
refresh agreement high, but it did not produce calculator-result discovery.

## Next

Do not retune simple parameter-delta caps `0.05` or `0.10` on this setup as
novelty. Next work should try a trust region that validates per-step
improvement, hard/assignment-style usage constraints, Jacobian-conditioned
state, or richer targets.
