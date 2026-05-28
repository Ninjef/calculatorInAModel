# 2026-05-28 Online Shadow Feedback Target Prototype Gate

## Question

Can a fit-split average boundary-gradient target per boundary-best result
class stabilize the online MLP shadow target enough to pass the heldout
model-gradient gate?

## Implementation

- Added a boundary-gradient helper that also returns the boundary-best result
  class for each example.
- Added `fit_shadow_feedback_target_prototypes`.
- Added `--shadow-feedback-target-transform fit_result_prototype`.
- The transform fits prototypes on the fit split only, applies them by class
  on validation and heldout examples, and falls back to the raw target only for
  classes missing from the fit split.
- Final diagnostics still compare induced model gradients against the original
  boundary-target gradients.

## Runs

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_prototype_gate
runs/2026-05-28_phase7_online_shadow_feedback_target_prototype_gap_selection_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `online_mlp` shadow diagnostic only.
- heldout fraction `0.2`, validation fraction `0.1`.
- feature mode `injection_grad_logits`.
- target normalization `fit_zscore_per_result`.
- target transform `fit_result_prototype`.
- learning rate `0.001`, warmup `100`, updates per step `1`.

Results:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cosine` | `16` | `90` | `0.77715 / 0.80938` | `0.18232 / 0.14664` | `1.1946 / 1.0812` |
| `cosine` | `32` | `80` | `0.80402 / 0.82434` | `0.19088 / 0.15571` | `1.0286 / 1.0208` |
| `mse_plus_cosine` | `16` | `100` | `0.78792 / 0.81720` | `0.19386 / 0.15637` | `1.1355 / 1.0696` |
| `mse_plus_cosine` | `32` | `90` | `0.79237 / 0.82132` | `0.20187 / 0.15964` | `1.0651 / 1.0565` |

Gap-selection follow-up:

| Hidden | Loss mode | Gap penalty | Step | Heldout result/upstream cosine | Train-heldout gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| `16` | `cosine` | `3.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` |
| `16` | `cosine` | `4.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` |
| `16` | `cosine` | `5.0` | `80` | `0.75397 / 0.78550` | `0.17047 / 0.14094` |
| `32` | `cosine` | `4.0` | `80` | `0.80402 / 0.82434` | `0.19088 / 0.15571` |

## Conclusion

```text
online_mlp_shadow_feedback_target_prototype_partial_no_go
```

Prototype averaging improves heldout direction but not enough. The best h32
run crosses `0.80` heldout result cosine, but gaps remain too high. The best
h16 gap-selected run still has result gap `0.1705`.

## Anti-Regression Note

Do not keep circling prototype targets or selection penalties on this same
state/objective. The durable blocker is the fit-heldout result-gradient gap,
so the next branch should change learned-gradient state or train directly
against a gap/norm objective.
