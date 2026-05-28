# 2026-05-28 Non-Bottleneck Causal Gap Gate

## Question

Can a zero-injection causal-use hinge make additive non-bottleneck hard
assignment learn real calculator use?

## Code Change

Added:

- `--calculator-causal-gap-weight`
- `--calculator-causal-gap-margin`

The objective computes `zero_injection_loss - normal_loss` and applies a hinge
requiring the zero-injection path to be worse by the configured margin. The
training curve logs the gap, zero loss, normal loss, and objective value.

## Runs

Run root:

```text
runs/2026-05-28_phase7_non_bottleneck_causal_gap_gate
```

Shared configuration:

- model-c, natural `0..19`, exact-grid batch.
- `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `calculator_action_head=result_space`.
- `answer_loss_weight=1`.
- `result_policy_improvement_assignment_weight=10`.
- `calculator_causal_gap_margin=0.5`.
- 800 steps, snapshots every `50`.

## Results

| Setup | Final eval exact | Best normal snapshot | Last zero-injection | Last learned calc | Best result-policy acc | Last causal gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| assignment `10`, no gap | `0.700` | `0.8200` at `650` | `0.6500` | `0.0325` | `0.0575` | n/a |
| assignment `10`, gap weight `10` | `0.560` | `0.5750` at `750` | `0.3375` | `0.0000` | `0.0300` | `1.2717` |
| assignment `10`, gap weight `50` | `0.4225` | `0.4800` at `750` | `0.2750` | `0.0425` | `0.0450` | `0.8372` |

## Conclusion

```text
non_bottleneck_causal_gap_pressure_negative
```

The hinge successfully creates an ablation loss gap, but it does not align the
result policy with true sums. In this setup, causal pressure mostly hurts the
zero-injection path and lowers answer accuracy.

## Anti-Regression Note

Do not repeat additive result-space `ste` with assignment weight `10`, causal
gap margin `0.5`, weights `10/50`, and 800-step CLI seed `2` as novelty. The
next non-bottleneck attempt needs staged bottleneck-to-additive transfer or a
causal target tied to correct calculator utility.
