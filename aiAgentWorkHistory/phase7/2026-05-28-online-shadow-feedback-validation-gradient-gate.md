# 2026-05-28 Online Shadow Feedback Validation-Gradient Gate

## Question

Can the online MLP shadow module clear the heldout warmup gate if it is trained
against the actual model-gradient object on the validation split, not only the
per-example result-logit target?

## Implementation

- Added `--shadow-feedback-validation-gradient-loss-weight`.
- Added `--shadow-feedback-validation-gradient-norm-weight`.
- Added differentiable grouped model-gradient comparison for result-proj and
  upstream parameter groups.
- Split shadow feature extraction from target construction.
- Enabled fixed online-MLP shadow feedback for Stage 1:
  - fit the shadow module once before training;
  - save `online_shadow_feedback_module.pt`;
  - train the model with feature-only fixed shadow feedback, without boundary
    target recomputation in the training loop.

## Runs

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate/stage1_online_shadow_feedback_early_lift
runs/2026-05-28_phase7_online_shadow_feedback_validation_gradient_gate/stage1_online_shadow_feedback_weight_sweep
```

Common Stage 0B configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `online_mlp` shadow diagnostic.
- heldout fraction `0.2`, validation fraction `0.1`.
- feature mode `injection_grad_logits`.
- target normalization `fit_zscore_per_result`.
- loss mode `cosine`.
- learning rate `0.001`, warmup `100`, updates per step `1`.
- validation-gradient loss weight `0.5`.

Stage 0B results:

| Hidden | Norm weight | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `0.0` | `100` | `0.81317 / 0.81645` | `0.07823 / 0.04347` | `1.5099 / 1.3677` |
| `16` | `0.1` | `100` | `0.80523 / 0.81281` | `0.09737 / 0.04914` | `1.3299 / 1.2108` |
| `32` | `0.0` | `100` | `0.80489 / 0.80824` | `0.12059 / 0.13201` | `1.1647 / 1.1000` |
| `32` | `0.1` | `100` | `0.80683 / 0.80826` | `0.12274 / 0.13430` | `1.1276 / 1.0736` |

Stage 1 fixed-module early-lift smoke:

| Shadow weight | Final exact match | Best snapshot exact | Notes |
| ---: | ---: | ---: | --- |
| `1.0` | `0.075` | `0.0525` | shadow norm grew to `79123.875` |
| `0.01` | `0.005` | `0.0400` | shadow norm grew to `73802.805` |
| `0.001` | `0.035` | `0.0550` | shadow norm grew to `64064.488` |

## Conclusion

```text
online_mlp_shadow_feedback_validation_gradient_stage0b_pass_stage1_fixed_module_negative
```

Direct validation model-gradient regularization produced the first clean
online-shadow Stage 0B pass. But freezing that calibrated module for Stage 1
does not work: it remains below the output-projection feedback baseline and
the feedback norm explodes as model features move out of distribution.

## Anti-Regression Note

Do not rerun the same h16/h32 validation-gradient `0.5`, norm `0/0.1` Stage
0B grid or the fixed-module Stage 1 weights `1.0/0.01/0.001` as novelty. The
next branch should make the shadow module on-policy during Stage 1 or clamp
the feedback update so the Stage 0B signal remains valid after model movement.
