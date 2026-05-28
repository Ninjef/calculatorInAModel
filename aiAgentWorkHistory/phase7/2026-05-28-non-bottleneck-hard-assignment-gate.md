# 2026-05-28 Non-Bottleneck Hard Assignment Gate

## Question

Can the hard improvement-assignment objective train calculator use when the
calculator is only one additive path through the model and the normal neuron
path remains available?

## Code Change

Allowed `calculator_action_head=result_space` with `calculator_estimator=ste`.
This lets the result-space head run in ordinary additive mode without the
strict answer decoder, and makes a clean answer-only baseline possible.

## Runs

Run root:

```text
runs/2026-05-28_phase7_non_bottleneck_hard_assignment_gate
```

Shared configuration:

- model-c, natural `0..19`, exact-grid batch.
- `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `calculator_action_head=result_space`.
- `answer_loss_weight=1`.
- 800 steps, snapshots every `50`.

## Results

| Setup | Final eval exact | Best normal snapshot | Last normal snapshot | Last learned calc | Best result-policy acc | Last assignment target acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer loss only | `0.615` | `0.9725` at `600` | `0.6325` | `0.0250` | n/a | n/a |
| answer loss + assignment `10` | `0.700` | `0.8200` at `650` | `0.6775` | `0.0325` | `0.0575` | `0.0033` |

Selected controls:

- Answer-only best snapshot: normal `0.9725`, injection-zero `0.560`,
  oracle `0.520`, learned calculator result `0.035`.
- Assignment best snapshot: normal `0.820`, injection-zero `0.740`, oracle
  `0.7375`, learned calculator result `0.0275`.
- Assignment final snapshot: normal `0.6775`, injection-zero `0.650`,
  oracle `0.555`, learned calculator result `0.0325`.

## Conclusion

```text
non_bottleneck_hard_assignment_transfer_negative
```

The additive model can solve substantially through the neuron path. Hard
improvement assignment does not force useful calculator use under bypass: the
learned result policy stays near chance and the assignment targets themselves
become mostly wrong.

## Anti-Regression Note

Do not repeat the same additive result-space `ste`, answer-loss `1`,
assignment-weight `10`, 800-step CLI seed `2` gate as novelty. The next
non-bottleneck attempt should add causal-use pressure, a staged
bottleneck-to-additive handoff, or a target construction robust to bypass.
