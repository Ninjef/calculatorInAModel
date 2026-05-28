# Phase 7 Thirty-Second Task: Answer-Loss Line Search Gate

## Purpose

Test whether refreshed online-shadow Stage 1 can learn if each proposed
optimizer step is repaired by a hard-path answer-loss line search over scaled
versions of the proposed parameter delta.

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
- Acceptance mode: `answer_loss_line_search`.
- Line-search scales: `1,0.5,0.25,0.1,0`.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Run

| Scales | Accepted steps | Final exact | Best snapshot | Final learned calc |
| --- | ---: | ---: | ---: | ---: |
| `1,0.5,0.25,0.1,0` | `5/200` (`2.5%`) | `0.060` | `0.0925` | `0.0650` |

## Conclusion

```text
answer_loss_line_search_step_repair_stage1_negative
```

Line search over step size slightly improved the best snapshot versus plain
accept/reject, but almost every refreshed-shadow proposed step remained
locally harmful under hard-path answer loss.

## Next

Do not repeat these line-search scales on this refreshed h32
validation-gradient setup with feedback clamp `10` and a 200-step budget as
novelty. Next work should construct useful directions directly, or move to
hard/assignment-style usage constraints, Jacobian-conditioned state, or richer
targets.
