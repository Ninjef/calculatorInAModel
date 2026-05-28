# Phase 7 Twenty-Fourth Task: Online MLP Shadow Feedback Result-Input State Gate

## Purpose

Test whether the online MLP shadow-feedback module needs the calculator
result-projection input representation, not just answer-gradient and result
logit state.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits_result_input`.
- Target normalization: `fit_zscore_per_result`.
- Losses: `cosine`, `mse_plus_cosine`.
- Heldout: untouched `0.2` split.
- Validation: `0.1`, every `10` warmup steps.

## Results

| Loss mode | Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` | result-gap fail |
| `cosine` | `32` | `0.78949 / 0.82941` | `0.20794 / 0.15326` | `1.0190 / 1.0438` | gap fail |
| `mse_plus_cosine` | `16` | `0.74513 / 0.82490` | `0.19641 / 0.12421` | `1.2097 / 1.2548` | result-gap fail |
| `mse_plus_cosine` | `32` | `0.77821 / 0.82574` | `0.21862 / 0.15835` | `1.0516 / 1.0729` | gap fail |

Gap-penalized selection on h16/`cosine` with penalties `3/4/5` kept step
`100` and reproduced heldout `0.76756/0.83718` with gaps `0.19578/0.12689`.

## Conclusion

```text
online_mlp_shadow_feedback_result_input_state_negative
```

The result-input state improves upstream alignment but does not solve the
result-head train-heldout gap. No Stage 1 early-lift run was launched.

## Next

Do not rerun this raw result-input state as novelty. Next work should use
explicit train-time gap/norm penalties, Jacobian-conditioned state, or a
genuinely different learned-gradient target/state.
