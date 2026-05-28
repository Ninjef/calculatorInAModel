# 2026-05-28 Hard Improvement Assignment Gate

## Question

Can a hard assignment-style constraint prevent result collapse by tying diverse
result requests to actual per-example answer-loss improvement?

## Implementation

- Added `hard_improvement_assignment_targets`.
- Added `--result-policy-improvement-assignment-weight`.
- Added `--result-policy-improvement-assignment-min-improvement`.
- Added `--result-policy-improvement-assignment-quota-multiplier`.
- The assignment target compares the current learned result's forced answer
  loss against every forced result class, assigns only improving alternatives,
  and enforces a per-result quota.
- Result-policy stabilization can now be the active
  `direct_feedback_alignment` training objective without boundary or shadow
  feedback.

## Runs

Run root:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `answer_loss_weight=0`.
- assignment min improvement `0`.
- assignment quota multiplier `1.0`.
- 200 steps, snapshots every `25`.

Results:

| Setup | Assignment weight | Final exact | Best snapshot | Assigned fraction | Target accuracy | Hard effective results |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| refreshed h32 shadow, clamp `10` | `1` | `0.0475` | `0.0650` | `0.9250` | `0.8189` | `1.00` |
| refreshed h32 shadow, clamp `10` | `10` | `0.1700` | `0.2425` | `0.7925` | `0.9117` | `14.12` |
| no shadow feedback | `10` | `0.4000` | `0.3500` | `0.6175` | `0.9474` | `18.85` |

## Conclusion

```text
hard_improvement_assignment_stage1_lift_partial
```

Hard assignment pressure is a promising teaching signal. It beats the old
`0.16` early-lift baseline and avoids the one-result collapse when weighted
strongly enough. The no-shadow ablation shows the assignment target, not the
shadow module, is the main active ingredient.

## Anti-Regression Note

Do not repeat the same seed-2/seed-4 exact-grid 200-step assignment weights
`1` or `10` as novelty. This branch now needs retention, seeds, longer
convergence, and cheaper/scalable approximations.
