# 2026-05-28 Hard Improvement Assignment Retention Gate

## Question

Does a result interface taught by hard improvement assignment survive after
the assignment objective decays away and only natural answer loss remains?

## Run

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_retention_gate/decay200_answer1
```

Configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- no shadow feedback.
- `result_policy_improvement_assignment_weight=10`.
- `result_policy_stabilization_decay_steps=200`.
- `answer_loss_weight=1`.
- 400 steps, snapshots every `25`.

## Results

| Step | Assignment weight | Snapshot exact | Result-policy accuracy | Hard effective results |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `10.0` | `0.0250` | `0.0225` | `6.72` |
| `100` | `5.0` | `0.2700` | `0.2650` | `18.30` |
| `175` | `1.25` | `0.3700` | n/a | n/a |
| `200` | `0.0` | `0.3475` | `0.3575` | `18.54` |
| `250` | `0.0` | `0.1050` | `0.0975` | `8.78` |
| `400` | `0.0` | `0.1050` | `0.0975` | `8.73` |

Final eval exact was `0.1075`.

## Conclusion

```text
hard_improvement_assignment_decay_retention_negative
```

Plain answer loss did not retain the assignment-taught interface. The result
policy collapsed soon after the assignment term reached zero.

## Anti-Regression Note

Do not repeat assignment weight `10 -> 0` over 200 steps with
`answer_loss_weight=1`, no shadow feedback, and 400-step budget as novelty.
