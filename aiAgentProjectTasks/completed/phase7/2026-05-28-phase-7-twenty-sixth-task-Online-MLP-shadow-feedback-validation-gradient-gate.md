# Phase 7 Twenty-Sixth Task: Online MLP Shadow Feedback Validation-Gradient Gate

## Purpose

Test whether direct train-time validation model-gradient regularization can
close the online MLP shadow train-heldout gap, then check whether the passing
module produces early Stage 1 lift.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Features: `injection_grad_logits`.
- Target normalization: `fit_zscore_per_result`.
- Prediction loss: `cosine`.
- Heldout: untouched `0.2` split.
- Validation: `0.1`, every `10` warmup steps.
- Validation-gradient loss weight: `0.5`.
- Validation-gradient norm weights: `0.0`, `0.1`.

## Stage 0B Results

| Hidden | Norm weight | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `0.0` | `0.81317 / 0.81645` | `0.07823 / 0.04347` | `1.5099 / 1.3677` | norm concern |
| `16` | `0.1` | `0.80523 / 0.81281` | `0.09737 / 0.04914` | `1.3299 / 1.2108` | pass but high norm |
| `32` | `0.0` | `0.80489 / 0.80824` | `0.12059 / 0.13201` | `1.1647 / 1.1000` | pass |
| `32` | `0.1` | `0.80683 / 0.80826` | `0.12274 / 0.13430` | `1.1276 / 1.0736` | pass |

## Stage 1 Results

Fixed calibrated h32/norm `0.1` module, 200-step early-lift smoke:

| Shadow weight | Final exact match | Best snapshot exact |
| ---: | ---: | ---: |
| `1.0` | `0.075` | `0.0525` |
| `0.01` | `0.005` | `0.0400` |
| `0.001` | `0.035` | `0.0550` |

## Conclusion

```text
online_mlp_shadow_feedback_validation_gradient_stage0b_pass_stage1_fixed_module_negative
```

The direct validation-gradient objective is a real Stage 0B improvement, but
the fixed calibrated module does not produce useful early Stage 1 lift. It
falls below the `0.16` output-projection boundary-feedback baseline and shows
shadow norm blow-up as the model moves.

## Next

Do not rerun this fixed-module weight sweep as novelty. Next work should use
periodic on-policy shadow refresh, trust-region/norm-clamped feedback, or
state/targets that remain valid after upstream movement.
