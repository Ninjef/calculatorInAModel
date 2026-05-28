# Phase 7 Thirty-Third Task: Output-Jacobian Shadow Feature Gate

## Purpose

Test whether an explicitly Jacobian-conditioned online shadow state improves
heldout gradient agreement and refreshed Stage 1 discovery.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- New feature mode: `injection_grad_logits_output_jacobian`.
- Feature definition: answer-loss injection gradient, result logits, and
  local `J_output^T answer_grad` scores from the calculator output projection.
- Stage 0B: validation-gradient weight `0.5`, norm weight `0.1`.
- Stage 1: h32 feature-normalized module, refresh every `50`, feedback clamp
  `10`, 200-step early-lift smoke.

## Stage 0B Runs

| Hidden | Feature norm | Heldout result/upstream cosine | Train-heldout gap |
| ---: | --- | ---: | ---: |
| `16` | none | `0.6703 / 0.7245` | `0.1013 / 0.0938` |
| `32` | none | `0.7957 / 0.8237` | `0.0994 / 0.1079` |
| `32` | `fit_zscore_per_feature` | `0.9073 / 0.9011` | `0.0639 / 0.0736` |

## Stage 1 Run

| Final exact | Best snapshot | Final learned calc | Refresh result cosines |
| ---: | ---: | ---: | --- |
| `0.055` | `0.065` | `0.0475` | `0.9073`, `0.9982`, `0.9988`, `0.9988`, `0.9990` |

## Conclusion

```text
output_jacobian_shadow_feature_stage0b_pass_stage1_negative
```

The output-Jacobian feature is useful for the heldout gradient diagnostic, but
state conditioning alone did not fix refreshed Stage 1 dynamics.

## Next

Do not repeat this h32 feature-normalized output-Jacobian feature with
validation-gradient `0.5`, norm `0.1`, refresh every `50`, clamp `10`, and
200-step budget as novelty. Next work should move to hard assignment-style
usage constraints, richer targets, or a learned update path that constructs
better directions.
