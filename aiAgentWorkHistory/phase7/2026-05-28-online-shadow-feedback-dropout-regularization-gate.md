# 2026-05-28 Online Shadow Feedback Dropout Regularization Gate

## Question

Can ordinary dropout regularization rescue the directional-loss online MLP
shadow-feedback branch by reducing train-heldout gradient overfit?

## Implementation

- Added `shadow_feedback_dropout` and `shadow_feedback_weight_decay` to
  `TrainConfig`.
- Added dropout support to `ShadowFeedbackMLP`.
- Switched the online shadow optimizer to use explicit `AdamW` weight decay.
- Added CLI/config plumbing and tests for both knobs.

## Runs

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_dropout_gate
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
- weight decay `0.01`.

Results:

| Hidden | Dropout | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `0.1` | `100` | `0.77227 / 0.80618` | `0.20170 / 0.15624` | `1.1453 / 1.0777` |
| `16` | `0.2` | `100` | `0.76423 / 0.79834` | `0.19773 / 0.15298` | `1.1706 / 1.0851` |
| `32` | `0.1` | `100` | `0.79199 / 0.82479` | `0.20391 / 0.15643` | `1.0226 / 1.0241` |
| `32` | `0.2` | `100` | `0.79098 / 0.81871` | `0.20355 / 0.16115` | `1.0373 / 1.0340` |

## Conclusion

```text
online_mlp_shadow_feedback_dropout_regularization_no_go
```

Dropout does not fix the overfit mode. It keeps heldout cosines high, but the
result train-heldout gap stays near `0.20`, above the `0.15` gate. This is too
weak to justify Stage 1.

## Anti-Regression Note

This also closes the obvious "add dropout" branch for the current
target-normalized directional-loss logits-state setup. The repeated pattern is
now stable: selection, simple feature scaling, raw policy-state appending,
direction-only loss, gap-penalized selection, and dropout all fail because the
fit split remains easier than heldout. Future work should change the target
construction or the learned-gradient state itself, or optimize an explicit
train-time gap/norm objective.
