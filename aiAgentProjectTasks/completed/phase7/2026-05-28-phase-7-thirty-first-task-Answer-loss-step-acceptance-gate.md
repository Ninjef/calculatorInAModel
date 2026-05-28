# Phase 7 Thirty-First Task: Answer-Loss Step Acceptance Gate

## Purpose

Test whether refreshed online-shadow Stage 1 can learn if proposed optimizer
steps are accepted only when they improve hard-path answer loss.

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
- Acceptance mode: `answer_loss_decrease`.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Runs

| Tolerance | Accepted steps | Final exact | Best snapshot | Final learned calc |
| ---: | ---: | ---: | ---: | ---: |
| `0.0` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0475` |
| `0.1` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0450` |

## Conclusion

```text
answer_loss_step_acceptance_stage1_negative
```

Most proposed refreshed-shadow updates are locally harmful under hard-path
answer loss. Rejecting them stabilizes the run, but it does not produce
calculator-result discovery.

## Next

Do not repeat answer-loss acceptance with tolerances `0.0` or `0.1` on this
setup as novelty. Next work should repair or construct useful directions
rather than simply reject bad ones, or move to hard/assignment-style usage
constraints, Jacobian-conditioned state, or richer targets.
