# 2026-05-28 Optimizer Step Trust Region Gate

## Question

Is refreshed online-shadow Stage 1 failing because AdamW moves the model too
far between shadow refits?

## Implementation

- Added `--optimizer-step-max-delta-norm`.
- When enabled, the loop snapshots trainable parameters before
  `optimizer.step()`.
- After the step, it computes the actual parameter-update L2 norm and rescales
  the update if it exceeds the configured radius.
- Added training-curve metrics for actual delta, proposed delta, trust scale,
  and cap.

## Runs

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_optimizer_trust_region_gate
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

| Max delta | Final exact | Best snapshot | Min/median/last trust scale | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | `0.075` | `0.060` | `0.251 / 0.300 / 0.259` | `0.0475` | `5.70` |
| `0.10` | `0.040` | `0.045` | `0.550 / 0.624 / 0.594` | `0.0325` | `10.00` |

Refresh history:

| Max delta | Refresh heldout result cosines |
| ---: | --- |
| `0.05` | `0.9393`, `0.9774`, `0.8751`, `0.9711` |
| `0.10` | `0.9916`, `0.9889`, `0.9920`, `0.9939` |

## Conclusion

```text
optimizer_step_trust_region_stage1_negative
```

The simple parameter-distance trust region works mechanically and stabilizes
the refreshed-shadow run, but it still does not align examples to useful
calculator results.

## Anti-Regression Note

Do not repeat optimizer step caps `0.05` or `0.10` with this refreshed h32
validation-gradient module, clamp `10`, and 200-step budget as novelty.
