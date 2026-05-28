# Phase 7 Twenty-Ninth Task: Result-Policy Soft Diversity Gate

## Purpose

Test whether a non-prescriptive result-policy entropy/batch-diversity
stabilizer can prevent the refreshed online shadow Stage 1 run from collapsing
to a single calculator result.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Calibrated module: h32 validation-gradient module.
- Shadow weight: `1.0`.
- Refresh cadence: every `50` training steps.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Runs

| Entropy | Diversity | Apply clamp | Final exact | Best snapshot | Final hard effective results |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | `1.0` | none | `0.015` | `0.0475` | `1.00` |
| `0.01` | `1.0` | `10` | `0.005` | `0.0400` | `1.00` |
| `0.0` | `100.0` | `10` | `0.070` | `0.0800` | `9.14` |

## Conclusion

```text
result_policy_soft_diversity_stabilization_stage1_negative
```

Low soft batch-diversity did not stop hard result collapse. High diversity
plus clamp did keep broader hard usage, but it still did not align examples
to useful calculator results and remained below the `0.16` boundary-feedback
baseline.

## Next

Do not retune this soft marginal entropy branch as novelty. Next work should
try a hard/assignment-style usage constraint, step-level trust region,
Jacobian-conditioned state, or richer target that links diverse requests to
per-example improvement.
