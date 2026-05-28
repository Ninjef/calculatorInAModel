# 2026-05-28 Answer-Loss Step Acceptance Gate

## Question

Can refreshed online-shadow Stage 1 learn if we reject proposed optimizer
steps that worsen real hard-path answer loss?

## Implementation

- Added `--optimizer-step-acceptance-mode`.
- Added `--optimizer-step-acceptance-tolerance`.
- `answer_loss_decrease` snapshots trainable parameters before
  `optimizer.step()`, evaluates hard-path answer loss after the step, and
  restores the snapshot if the loss increase exceeds tolerance.
- Added cumulative acceptance metrics to `training_curve.csv`.

## Runs

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_answer_loss_acceptance_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- h32 validation-gradient online shadow module.
- `shadow_feedback_weight=1.0`.
- `shadow_feedback_apply_max_norm=10`.
- refresh every `50` steps.
- 200 steps, snapshots every `25`.

Results:

| Tolerance | Accepted steps | Final exact | Best snapshot | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0475` | `3.12` |
| `0.1` | `6/200` (`3%`) | `0.050` | `0.070` | `0.0450` | `3.12` |

Refresh heldout result cosines stayed usable:

| Tolerance | Refresh heldout result cosines |
| ---: | --- |
| `0.0` | `0.8526`, `0.8313`, `0.7288`, `0.8594` |
| `0.1` | `0.8526`, `0.8313`, `0.7526`, `0.8642` |

## Conclusion

```text
answer_loss_step_acceptance_stage1_negative
```

The acceptance gate shows that most refreshed-shadow proposed steps are not
locally useful under the real hard answer-loss surface. Filtering them is a
diagnostic, not a rescue.

## Anti-Regression Note

Do not repeat answer-loss step acceptance with tolerances `0.0` or `0.1`,
refreshed h32 validation-gradient module, clamp `10`, and 200-step budget as
novelty.
