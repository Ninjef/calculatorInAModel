# Phase 7 Thirty-Seventh Task: Non-Bottleneck Hard Assignment Gate

## Purpose

Test whether the hard improvement-assignment signal transfers from the strict
answer-decoder bottleneck to the additive non-bottleneck setting, where a
normal neuron path can also solve the task.

## Setup

- Base task: natural `0..19` exact-grid, model-c, CLI seed `2`.
- Bottleneck: none; additive calculator injection into the normal residual
  stream.
- Estimator: `ste`.
- Action head: `result_space`.
- Answer loss weight: `1`.
- Assignment comparison: weight `0` baseline vs weight `10`.
- Training: 800 steps, snapshots every `50`.

## Code Change

Allowed `calculator_action_head=result_space` with `calculator_estimator=ste`
so non-bottleneck result-space baselines and assignment runs can execute
without the strict answer decoder.

## Result

| Setup | Final eval exact | Best normal snapshot | Last normal snapshot | Last learned calc | Best result-policy acc | Last assignment target acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer loss only | `0.615` | `0.9725` at `600` | `0.6325` | `0.0250` | n/a | n/a |
| answer loss + assignment `10` | `0.700` | `0.8200` at `650` | `0.6775` | `0.0325` | `0.0575` | `0.0033` |

High injection-zero accuracy showed bypass in both runs. At the assignment
run's best normal snapshot, injection-zero was `0.740` and oracle was `0.7375`.

## Conclusion

```text
non_bottleneck_hard_assignment_transfer_negative
```

The bottleneck hard-assignment signal does not transfer directly to the
additive non-bottleneck model. The model improves answer accuracy mostly
through the neuron path, while calculator-result accuracy remains near chance.

## Next

Do not rerun this exact additive `ste`, assignment-weight-`10`, 800-step seed
as novelty. Future non-bottleneck work needs explicit causal calculator-use
pressure, staged bottleneck-to-additive handoff, or an improvement target that
stays valid when bypass is available.
