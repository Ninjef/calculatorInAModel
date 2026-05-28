# 2026-05-28 Output-Jacobian Shadow Feature Gate

## Question

Does adding local calculator output-Jacobian information to the online shadow
state improve gradient agreement and Stage 1 calculator-result discovery?

## Implementation

- Added `injection_grad_logits_output_jacobian` to
  `--shadow-feedback-feature-mode`.
- The mode appends `J_output^T answer_grad` scores to the existing
  answer-gradient plus result-logit state.
- Added `shadow_feedback_feature_output_jacobian_l2` metrics.
- Added a unit test verifying the appended slice equals the calculator output
  projection transpose times the scaled answer-loss injection gradient.

## Runs

Run root:

```text
runs/2026-05-28_phase7_output_jacobian_shadow_feature_gate
```

Stage 0B diagnostics:

| Hidden | Feature norm | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| ---: | --- | ---: | ---: | ---: |
| `16` | none | `0.6703 / 0.7245` | `0.1013 / 0.0938` | `1.8176 / 1.7362` |
| `32` | none | `0.7957 / 0.8237` | `0.0994 / 0.1079` | `1.2553 / 1.1170` |
| `32` | `fit_zscore_per_feature` | `0.9073 / 0.9011` | `0.0639 / 0.0736` | `1.3044 / 1.2598` |

Stage 1 refreshed h32 feature-normalized smoke:

| Final exact | Best snapshot | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: |
| `0.055` | `0.065` | `0.0475` | `10.00` |

Refresh heldout result/upstream cosines:

| Step | Result cosine | Upstream cosine | Result gap |
| ---: | ---: | ---: | ---: |
| `0` | `0.9073` | `0.9011` | `0.0639` |
| `50` | `0.9982` | `1.0000` | `0.0018` |
| `100` | `0.9988` | `1.0000` | `0.0010` |
| `150` | `0.9988` | `1.0000` | `0.0011` |
| `200` | `0.9990` | `1.0000` | `0.0009` |

## Conclusion

```text
output_jacobian_shadow_feature_stage0b_pass_stage1_negative
```

The Jacobian-conditioned feature state is a real Stage 0B improvement after
feature z-scoring, but it does not solve Stage 1. The model still fails to
turn refreshed local gradient agreement into learned calculator-result use.

## Anti-Regression Note

Do not repeat h16/h32 raw output-Jacobian feature diagnostics or h32
feature-normalized refreshed Stage 1 with clamp `10`, refresh `50`, and
200-step budget as novelty.
