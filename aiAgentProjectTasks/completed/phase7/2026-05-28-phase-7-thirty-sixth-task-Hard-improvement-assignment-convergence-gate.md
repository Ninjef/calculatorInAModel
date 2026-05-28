# Phase 7 Thirty-Sixth Task: Hard Improvement Assignment Convergence Gate

## Purpose

Test whether hard improvement assignment keeps improving when left on for a
longer budget, and whether that convergence behavior replicates across seeds.

## Setup

- Base task: natural `0..19` exact-grid, model-c.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow feedback: none.
- Assignment weight: `10`.
- Assignment min improvement: `0`.
- Assignment quota multiplier: `1`.
- Training: 800 or 1600 steps, snapshots every `50` or `100`.

## Result

| CLI seed | Steps | Answer loss weight | Final eval exact | Best snapshot | Last snapshot | Last result-policy acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `800` | `0` | `0.860` | `0.8525` at `800` | `0.8525` | `0.8350` |
| `2` | `800` | `1` | `0.860` | `0.8525` at `800` | `0.8525` | `0.8350` |
| `2` | `1600` | `0` | `0.915` | `0.9475` at `1300` | `0.9150` | `0.9025` |
| `4` | `1600` | `0` | `0.860` | `0.8700` at `1600` | `0.8700` | `0.8450` |
| `5` | `1600` | `0` | `0.820` | `0.9200` at `1500` | `0.8325` | `0.8250` |

Oracle stayed `1.000` in the final snapshots, injection-zero stayed near
chance, and operand exact stayed low. The learned result-space calculator path
is doing the work.

## Conclusion

```text
hard_improvement_assignment_convergence_seed_replication_mixed_partial
```

Always-on hard improvement assignment can train the natural result-space
calculator interface from scratch across seeds. It is not solved: one seed
drifted down after a strong peak, the target-off decay gate failed, and the
assignment target still scores forced result classes during training.

## Next

Do not rerun the same always-on 800 or 1600 step seed set as novelty. Next
work should test cheaper assignment construction, stronger handoff/retention,
stability selection, or the non-bottleneck setting.
