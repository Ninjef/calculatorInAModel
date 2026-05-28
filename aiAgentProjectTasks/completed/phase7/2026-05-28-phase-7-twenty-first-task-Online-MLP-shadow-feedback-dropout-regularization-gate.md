# Phase 7 Twenty-First Task: Online MLP Shadow Feedback Dropout Regularization Gate

## Purpose

Test whether ordinary training-time regularization closes the directional-loss
online MLP shadow-feedback generalization gap without changing the target or
input state.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits`.
- Target normalization: `fit_zscore_per_result`.
- Loss: `cosine`.
- Selection: ordinary validation `min_result_upstream_cosine`.
- Heldout: untouched `0.2` split.
- New knobs: `--shadow-feedback-dropout` and
  `--shadow-feedback-weight-decay`.

## Results

| Hidden | Dropout | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.1` | `0.77227 / 0.80618` | `0.20170 / 0.15624` | `1.1453 / 1.0777` | gap fail |
| `16` | `0.2` | `0.76423 / 0.79834` | `0.19773 / 0.15298` | `1.1706 / 1.0851` | gap fail |
| `32` | `0.1` | `0.79199 / 0.82479` | `0.20391 / 0.15643` | `1.0226 / 1.0241` | gap fail |
| `32` | `0.2` | `0.79098 / 0.81871` | `0.20355 / 0.16115` | `1.0373 / 1.0340` | gap fail |

## Conclusion

```text
online_mlp_shadow_feedback_dropout_regularization_no_go
```

Dropout preserves the useful directional-loss heldout cosine, especially for
h32, but does not close the train-heldout result-gradient gap. No Stage 1
early-lift run was launched.

## Next

Do not rerun dropout-only h16/h32 sweeps on this same state/objective as
novelty. Next work should change target construction or learned-gradient
state, or add explicit training-time gap/norm penalties.
