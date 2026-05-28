# Phase 7 Twenty-Third Task: Online MLP Shadow Feedback Target Prototype Gate

## Purpose

Test whether structural target stabilization by boundary-result prototypes can
reduce online MLP shadow-feedback overfit while preserving the useful
directional signal.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits`.
- Target normalization: `fit_zscore_per_result`.
- Target transform: `fit_result_prototype`.
- Losses: `cosine`, `mse_plus_cosine`.
- Heldout: untouched `0.2` split.
- Validation: `0.1`, every `10` warmup steps.

## Results

| Loss mode | Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `0.77715 / 0.80938` | `0.18232 / 0.14664` | `1.1946 / 1.0812` | result-gap fail |
| `cosine` | `32` | `0.80402 / 0.82434` | `0.19088 / 0.15571` | `1.0286 / 1.0208` | gap fail |
| `mse_plus_cosine` | `16` | `0.78792 / 0.81720` | `0.19386 / 0.15637` | `1.1355 / 1.0696` | gap fail |
| `mse_plus_cosine` | `32` | `0.79237 / 0.82132` | `0.20187 / 0.15964` | `1.0651 / 1.0565` | gap fail |

Gap-penalized selection follow-up on h16/cosine selected step `80` for
penalties `3/4/5`, reaching heldout `0.7540/0.7855` with gaps
`0.1705/0.1409`. This still missed the result-gap gate.

## Conclusion

```text
online_mlp_shadow_feedback_target_prototype_partial_no_go
```

Prototype targets slightly improve the tradeoff and produce the best heldout
result cosine in this online MLP branch, but they do not close the
train-heldout result gap enough to justify Stage 1.

## Next

Do not rerun this prototype-target setup or its gap-selection penalties as
novelty. Next work should change learned-gradient state or add explicit
train-time gap/norm penalties.
