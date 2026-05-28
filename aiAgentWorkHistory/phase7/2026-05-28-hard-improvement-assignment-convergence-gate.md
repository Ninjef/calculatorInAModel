# 2026-05-28 Hard Improvement Assignment Convergence Gate

## Question

Does the hard improvement-assignment branch keep climbing with a longer
always-on budget, and does the lift replicate across seeds?

## Run

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_convergence_gate
```

Shared configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- no shadow feedback.
- `result_policy_improvement_assignment_weight=10`.
- `result_policy_improvement_assignment_min_improvement=0`.
- `result_policy_improvement_assignment_quota_multiplier=1`.
- 800 or 1600 steps.

## Results

| CLI seed | Steps | Answer loss weight | Final eval exact | Best snapshot | Last snapshot | Last result-policy acc | Final learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `800` | `0` | `0.860` | `0.8525` at `800` | `0.8525` | `0.8350` | `0.8525` |
| `2` | `800` | `1` | `0.860` | `0.8525` at `800` | `0.8525` | `0.8350` | `0.8525` |
| `2` | `1600` | `0` | `0.915` | `0.9475` at `1300` | `0.9150` | `0.9025` | `0.9150` |
| `4` | `1600` | `0` | `0.860` | `0.8700` at `1600` | `0.8700` | `0.8450` | `0.8700` |
| `5` | `1600` | `0` | `0.820` | `0.9200` at `1500` | `0.8325` | `0.8250` | `0.8325` |

The 800-step `answer_loss_weight=0` and `1` curves were numerically identical
on the recorded metrics, so natural answer loss is not visibly contributing
to the discrete result policy while hard assignment remains active.

Final snapshot controls:

- oracle exact stayed `1.000`;
- injection-zero stayed near chance (`0.0375` to `0.0575`);
- operand exact stayed low (`0.0675` to `0.0875`).

## Conclusion

```text
hard_improvement_assignment_convergence_seed_replication_mixed_partial
```

The hard assignment target is a real training signal for the natural
result-space calculator interface. It can reach high accuracy from scratch and
replicates materially across seeds, but it is not stable enough or scalable
enough to be the final method.

## Anti-Regression Note

Do not repeat the same no-shadow, assignment-weight-`10`, 800/1600-step runs
on CLI seeds `2/4/5` as novelty. The next useful tests are lower-cost
assignment construction, target-off handoff stronger than plain decay,
stability/checkpoint selection, and non-bottleneck training.
