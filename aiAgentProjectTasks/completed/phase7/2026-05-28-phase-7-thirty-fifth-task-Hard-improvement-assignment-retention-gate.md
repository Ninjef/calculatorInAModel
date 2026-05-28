# Phase 7 Thirty-Fifth Task: Hard Improvement Assignment Retention Gate

## Purpose

Test whether natural answer loss can retain a result interface learned from
hard improvement assignment after the assignment target decays to zero.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow feedback: none.
- Assignment weight: `10`.
- Assignment decay steps: `200`.
- Answer loss weight: `1`.
- Training: 400 steps, snapshots every `25`.

## Result

| Step | Assignment weight | Snapshot exact | Result-policy accuracy | Hard effective results |
| ---: | ---: | ---: | ---: | ---: |
| `100` | `5.0` | `0.2700` | `0.2650` | `18.30` |
| `175` | `1.25` | `0.3700` | n/a | n/a |
| `200` | `0.0` | `0.3475` | `0.3575` | `18.54` |
| `250` | `0.0` | `0.1050` | `0.0975` | `8.78` |
| `400` | `0.0` | `0.1050` | `0.0975` | `8.73` |

Final eval exact: `0.1075`.

## Conclusion

```text
hard_improvement_assignment_decay_retention_negative
```

The assignment target teaches while present, but plain answer loss does not
retain the learned result interface after target-off.

## Next

Do not repeat this exact 200-step linear decay schedule as novelty. Next work
should test longer always-on convergence, seed replication, a stronger handoff
bridge, or lower-cost assignment approximations.
