# Phase 7 Thirty-Eighth Task: Non-Bottleneck Causal Gap Gate

## Purpose

Test whether a cheap, non-prescriptive causal-use pressure can rescue additive
non-bottleneck hard assignment.

## Setup

- Base task: natural `0..19` exact-grid, model-c, CLI seed `2`.
- Bottleneck: none; additive calculator injection into the normal residual
  stream.
- Estimator: `ste`.
- Action head: `result_space`.
- Answer loss weight: `1`.
- Assignment weight: `10`.
- Causal gap: hinge on `zero_injection_loss - normal_loss`.
- Causal gap margin: `0.5`.
- Causal gap weights: `10` and `50`.
- Training: 800 steps, snapshots every `50`.

## Code Change

Added `--calculator-causal-gap-weight` and
`--calculator-causal-gap-margin`. The training curve logs the causal gap,
normal loss, zero-injection loss, and objective value.

## Result

| Setup | Final eval exact | Best normal snapshot | Last zero-injection | Last learned calc | Best result-policy acc | Last causal gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| assignment `10`, no gap | `0.700` | `0.8200` at `650` | `0.6500` | `0.0325` | `0.0575` | n/a |
| assignment `10`, gap weight `10` | `0.560` | `0.5750` at `750` | `0.3375` | `0.0000` | `0.0300` | `1.2717` |
| assignment `10`, gap weight `50` | `0.4225` | `0.4800` at `750` | `0.2750` | `0.0425` | `0.0450` | `0.8372` |

## Conclusion

```text
non_bottleneck_causal_gap_pressure_negative
```

The causal-gap hinge can make the model depend on the calculator injection in
the narrow ablation sense, but it does not teach correct calculator-result
requests. It mostly damages the bypass path and lowers answer accuracy.

## Next

Do not repeat this exact margin-`0.5`, weight-`10/50`, assignment-weight-`10`,
800-step seed as novelty. Next non-bottleneck work should use a staged
bottleneck-to-additive handoff or a causal target tied to correct result-level
utility.
