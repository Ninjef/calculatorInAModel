# 2026-05-28 Answer-Loss Line Search Gate

## Question

Can refreshed online-shadow Stage 1 learn if we repair proposed optimizer
steps by trying several scaled versions and keeping only the best hard-path
answer-loss improvement?

## Implementation

- Added `--optimizer-step-line-search-scales`.
- Added `answer_loss_line_search` to `--optimizer-step-acceptance-mode`.
- The mode snapshots trainable parameters before the optimizer step, snapshots
  the proposed post-step parameters, evaluates configured scales of the fixed
  proposed delta, then applies the best improving scale or restores the
  pre-step parameters.
- Added training-curve fields for configured line-search scales and selected
  scale.

## Run

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_answer_loss_line_search_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- h32 validation-gradient online shadow module.
- `shadow_feedback_weight=1.0`.
- `shadow_feedback_apply_max_norm=10`.
- refresh every `50` steps.
- line-search scales `1,0.5,0.25,0.1,0`.
- 200 steps, snapshots every `25`.

Result:

| Accepted steps | Final exact | Best snapshot | Final learned calc | Final shadow norm |
| ---: | ---: | ---: | ---: | ---: |
| `5/200` (`2.5%`) | `0.060` | `0.0925` | `0.0650` | `3.28` |

Refresh heldout result cosines stayed usable:

| Refresh step | Heldout result cosine | Heldout upstream cosine | Result gap |
| ---: | ---: | ---: | ---: |
| `50` | `0.8819` | `0.9620` | `-0.0428` |
| `100` | `0.8611` | `0.9631` | `-0.0771` |
| `150` | `0.8361` | `0.9584` | `-0.0176` |
| `200` | `0.9037` | `0.9626` | `-0.0137` |

## Conclusion

```text
answer_loss_line_search_step_repair_stage1_negative
```

Line search is a small improvement over plain accept/reject, but it still does
not approach the `0.16` boundary-feedback baseline. The limiting issue is not
just step size: the proposed shadow directions are usually not useful under
the real hard answer-loss surface.

## Anti-Regression Note

Do not repeat answer-loss line search with scales `1,0.5,0.25,0.1,0`,
refreshed h32 validation-gradient module, clamp `10`, and 200-step budget as
novelty.
