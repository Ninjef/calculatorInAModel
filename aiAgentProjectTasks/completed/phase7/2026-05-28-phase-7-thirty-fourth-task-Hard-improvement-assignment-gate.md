# Phase 7 Thirty-Fourth Task: Hard Improvement Assignment Gate

## Purpose

Test whether a hard assignment-style usage constraint can link result diversity
to per-example answer-loss improvement and produce Stage 1 lift.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- New target: answer-loss-improving hard result assignment with per-result
  quota.
- Assignment min improvement: `0`.
- Assignment quota multiplier: `1.0`.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Runs

| Setup | Assignment weight | Final exact | Best snapshot | Final hard effective results |
| --- | ---: | ---: | ---: | ---: |
| refreshed h32 shadow, clamp `10` | `1` | `0.0475` | `0.0650` | `1.00` |
| refreshed h32 shadow, clamp `10` | `10` | `0.1700` | `0.2425` | `14.12` |
| no shadow feedback | `10` | `0.4000` | `0.3500` | `18.85` |

## Conclusion

```text
hard_improvement_assignment_stage1_lift_partial
```

Hard improvement assignment is the first recent mechanism to lift above the
`0.16` boundary-feedback baseline. The no-shadow ablation is strongest, so the
new target itself is doing most of the work.

## Next

Do not rerun the same 200-step weights `1` or `10` as novelty. Next test
longer convergence, target-off retention, seed replication, and cheaper
assignment approximations that do not score all result classes every step.
