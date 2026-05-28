# 2026-05-28 Online Shadow Feedback Result-Input State Gate

## Question

Can the online MLP shadow module generalize better if it receives the actual
calculator result-projection input representation?

## Implementation

- Added `calculator_read_result_logits_and_input`.
- Added `injection_grad_logits_result_input` feature mode.
- This feature mode concatenates per-example-scaled answer injection gradient,
  result logits, and the `result_proj` input representation.
- Added CLI/test coverage for the new feature mode and feature dimension.

## Runs

Run roots:

```text
runs/2026-05-28_phase7_online_shadow_feedback_result_input_state_gate
runs/2026-05-28_phase7_online_shadow_feedback_result_input_gap_selection_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- `online_mlp` shadow diagnostic only.
- heldout fraction `0.2`, validation fraction `0.1`.
- feature mode `injection_grad_logits_result_input`.
- target normalization `fit_zscore_per_result`.
- learning rate `0.001`, warmup `100`, updates per step `1`.

Results:

| Loss mode | Hidden | Step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cosine` | `16` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` | `1.1918 / 1.2309` |
| `cosine` | `32` | `100` | `0.78949 / 0.82941` | `0.20794 / 0.15326` | `1.0190 / 1.0438` |
| `mse_plus_cosine` | `16` | `100` | `0.74513 / 0.82490` | `0.19641 / 0.12421` | `1.2097 / 1.2548` |
| `mse_plus_cosine` | `32` | `100` | `0.77821 / 0.82574` | `0.21862 / 0.15835` | `1.0516 / 1.0729` |

Gap-selection follow-up:

| Hidden | Loss mode | Gap penalty | Step | Heldout result/upstream cosine | Train-heldout gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| `16` | `cosine` | `3.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` |
| `16` | `cosine` | `4.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` |
| `16` | `cosine` | `5.0` | `100` | `0.76756 / 0.83718` | `0.19578 / 0.12689` |

## Conclusion

```text
online_mlp_shadow_feedback_result_input_state_negative
```

Appending the result-projection input is useful for upstream direction but not
for result-head generalization. It leaves the same recurring result gap near
`0.20`, so it is not a Stage 1 go signal.

## Anti-Regression Note

Do not continue tuning raw result-input state, ordinary checkpoint selection,
or the already-tested h16/h32 directional losses as novelty. The next branch
should change the training objective itself or use a more structurally
different gradient state.
