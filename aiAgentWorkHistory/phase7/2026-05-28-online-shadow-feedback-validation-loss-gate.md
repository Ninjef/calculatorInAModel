# 2026-05-28 Online Shadow Feedback Validation-Loss Gate

## Question

Can the online MLP shadow module generalize better if the warmup objective
includes prediction loss on the validation split, not just fit-split loss?

## Implementation

- Added `shadow_feedback_validation_loss_weight` to `TrainConfig`.
- Added `--shadow-feedback-validation-loss-weight`.
- During online MLP shadow warmup, the fit loss can now be augmented with
  validation-split shadow prediction loss.
- The heldout split remains untouched for final diagnostic evaluation.
- Added summary metrics for validation-loss weight, final total objective, and
  final validation regularization objective.
- Added CLI and diagnostic test coverage.

## Runs

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_loss_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `online_mlp` shadow diagnostic only.
- heldout fraction `0.2`, validation fraction `0.1`.
- feature mode `injection_grad_logits`.
- target normalization `fit_zscore_per_result`.
- loss mode `cosine`.
- learning rate `0.001`, warmup `100`, updates per step `1`.

Results:

| Hidden | Validation-loss weight | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `0.5` | `70` | `0.74678 / 0.76350` | `0.17935 / 0.13529` | `1.2700 / 1.1495` |
| `16` | `1.0` | `60` | `0.72741 / 0.73810` | `0.15953 / 0.11499` | `1.3346 / 1.2494` |
| `32` | `0.5` | `100` | `0.79530 / 0.82329` | `0.19874 / 0.15688` | `0.9644 / 0.9768` |
| `32` | `1.0` | `100` | `0.79154 / 0.81952` | `0.19886 / 0.15922` | `0.9613 / 0.9919` |

## Conclusion

```text
online_mlp_shadow_feedback_validation_loss_regularization_no_go
```

Validation-loss regularization exposed the same tradeoff as earlier
checkpoint-selection and regularization branches. h32 keeps the heldout
direction signal but not the gap; h16 reduces the gap only by giving up
heldout/norm quality.

## Anti-Regression Note

Do not continue tuning ordinary validation prediction-loss weights on this
same h16/h32 directional-loss setup as novelty. The next branch should use a
direct split-gradient gap/norm objective, Jacobian-conditioned state, or a
structurally richer learned-gradient target.
