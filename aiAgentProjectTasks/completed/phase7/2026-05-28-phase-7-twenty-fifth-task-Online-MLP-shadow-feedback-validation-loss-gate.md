# Phase 7 Twenty-Fifth Task: Online MLP Shadow Feedback Validation-Loss Gate

## Purpose

Test whether adding validation-split prediction loss directly into the online
MLP shadow warmup objective closes the persistent train-heldout gradient gap.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits`.
- Target normalization: `fit_zscore_per_result`.
- Loss: `cosine`.
- Heldout: untouched `0.2` split.
- Validation: `0.1`, every `10` warmup steps.
- Validation-loss weights: `0.5`, `1.0`.

## Results

| Hidden | Validation-loss weight | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.5` | `0.74678 / 0.76350` | `0.17935 / 0.13529` | `1.2700 / 1.1495` | result-gap fail |
| `16` | `1.0` | `0.72741 / 0.73810` | `0.15953 / 0.11499` | `1.3346 / 1.2494` | tradeoff fail |
| `32` | `0.5` | `0.79530 / 0.82329` | `0.19874 / 0.15688` | `0.9644 / 0.9768` | gap fail |
| `32` | `1.0` | `0.79154 / 0.81952` | `0.19886 / 0.15922` | `0.9613 / 0.9919` | gap fail |

## Conclusion

```text
online_mlp_shadow_feedback_validation_loss_regularization_no_go
```

Train-time validation prediction loss is not enough. h32 keeps strong heldout
cosines but the result gap remains near `0.20`; h16 can reduce the gap only by
trading away heldout/norm quality. No Stage 1 early-lift run was launched.

## Next

Do not rerun ordinary validation-loss regularization as novelty. Next work
should use a direct split-gradient gap/norm objective, Jacobian-conditioned
state, or a richer learned-gradient target.
