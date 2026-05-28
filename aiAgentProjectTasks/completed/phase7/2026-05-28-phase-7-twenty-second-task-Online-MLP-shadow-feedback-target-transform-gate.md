# Phase 7 Twenty-Second Task: Online MLP Shadow Feedback Target Transform Gate

## Purpose

Test whether a lightweight target-stabilization transform can reduce online
MLP shadow-feedback overfit by removing per-example target magnitude before
the existing fit-split z-score normalization.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits`.
- Target normalization: `fit_zscore_per_result`.
- Target transform: `unit_norm_per_example`.
- Losses: `cosine`, `mse_plus_cosine`.
- Heldout: untouched `0.2` split.
- Validation: `0.1`, every `10` warmup steps.

## Results

| Loss mode | Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `0.76497 / 0.80097` | `0.19828 / 0.15457` | `1.2043 / 1.1108` | gap fail |
| `cosine` | `32` | `0.79363 / 0.82695` | `0.20246 / 0.15451` | `1.0244 / 1.0309` | gap fail |
| `mse_plus_cosine` | `16` | `0.77871 / 0.81139` | `0.20433 / 0.16068` | `1.1510 / 1.1032` | gap fail |
| `mse_plus_cosine` | `32` | `0.78550 / 0.81756` | `0.20889 / 0.16350` | `1.0733 / 1.0784` | gap fail |

## Conclusion

```text
online_mlp_shadow_feedback_target_unit_norm_no_go
```

Unit-normalizing each target row before z-scoring preserves the good heldout
direction signal but does not close the train-heldout gap. No Stage 1
early-lift run was launched.

## Next

Do not rerun this row-wise target-normalization branch as novelty. Next work
should use more structural target stabilization, a different learned-gradient
state, or explicit train-time gap/norm penalties.
