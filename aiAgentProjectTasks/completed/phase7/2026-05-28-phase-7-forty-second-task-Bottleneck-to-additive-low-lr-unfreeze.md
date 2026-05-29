# Phase 7 Forty-Second Task: Bottleneck-to-Additive Low-LR Unfreeze

## Status

Completed on 2026-05-28.

## Question

Can a low-LR full-policy unfreeze preserve or improve an adapted
non-bottleneck calculator handoff?

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_unfreeze_probe
```

Configuration:

- resumed from adapted weak-source additive checkpoints;
- `--semantic-decoder-checkpoint-load-scope full_model`;
- no `--freeze-calculator-policy`;
- global LR `3e-4`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

## Result

| Run | Final eval before | Final eval after | Last injection-zero | Last forced-random | Learned calc before | Learned calc after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` unfreeze | `0.6050` | `0.5200` | `0.0225` | `0.1200` | `0.8725` | `0.3000` |
| `src5_add5` unfreeze | `0.8175` | `0.8100` | `0.0225` | `0.1125` | `0.8000` | `0.2525` |

## Decision

```text
bottleneck_to_additive_low_lr_unfreeze_policy_collapse_negative
```

Plain low-LR full unfreeze damages the calculator policy. Future unfreezing
needs a retention constraint or a more selective schedule.
