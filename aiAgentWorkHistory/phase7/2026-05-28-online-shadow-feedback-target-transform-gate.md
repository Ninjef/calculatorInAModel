# 2026-05-28 Online Shadow Feedback Target Transform Gate

## Question

Is the directional-loss online MLP shadow module overfitting per-example target
magnitude rather than the gradient direction needed by the boundary ceiling?

## Implementation

- Added `shadow_feedback_target_transform` to `TrainConfig`.
- Added `--shadow-feedback-target-transform`.
- Added `unit_norm_per_example`, which normalizes each target-gradient row
  before fit-split target z-scoring.
- Threaded the transform through online MLP warmup, validation selection, and
  heldout gradient diagnostics.
- Added tests for transform behavior, CLI parsing, and diagnostic summary
  plumbing.

## Runs

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_transform_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `online_mlp` shadow diagnostic only.
- heldout fraction `0.2`, validation fraction `0.1`.
- feature mode `injection_grad_logits`.
- target normalization `fit_zscore_per_result`.
- target transform `unit_norm_per_example`.
- learning rate `0.001`, warmup `100`, updates per step `1`.

Results:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cosine` | `16` | `90` | `0.76497 / 0.80097` | `0.19828 / 0.15457` | `1.2043 / 1.1108` |
| `cosine` | `32` | `90` | `0.79363 / 0.82695` | `0.20246 / 0.15451` | `1.0244 / 1.0309` |
| `mse_plus_cosine` | `16` | `100` | `0.77871 / 0.81139` | `0.20433 / 0.16068` | `1.1510 / 1.1032` |
| `mse_plus_cosine` | `32` | `90` | `0.78550 / 0.81756` | `0.20889 / 0.16350` | `1.0733 / 1.0784` |

## Conclusion

```text
online_mlp_shadow_feedback_target_unit_norm_no_go
```

The transform changed the target construction but reproduced the same failure
mode as plain directional loss: heldout direction is high, yet result
train-heldout gap stays near `0.20`. No Stage 1 run was launched.

## Anti-Regression Note

Do not retest row-wise target norm removal as a new idea on this same
target-normalized logits-state setup. The next plausible branches are more
structural target averaging/prototypes, a different learned-gradient state, or
an explicit train-time gap/norm penalty.
