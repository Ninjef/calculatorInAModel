# 2026-05-28 Online Shadow-Feedback Feature-Standardization Gate

## Summary

Added fit-split per-feature z-score normalization to the online MLP
shadow-feedback diagnostic. The change tests whether the previous raw
policy-state and target-normalized near-miss branches were mainly failing from
ill-conditioned shadow input scales.

## Code

- Added `--shadow-feedback-feature-normalization`.
- Added `fit_zscore_per_feature`.
- Feature normalization stats are fit only on the fit split.
- Train, validation, and heldout features are normalized before the shadow MLP.
- Target predictions are still denormalized before raw model-gradient
  diagnostics.
- Added unit coverage for feature-normalizer fit-only stats and CLI defaults.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
100 passed
```

## Experiment

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_feature_norm_gate
```

Shared configuration:

- Shadow mode: `online_mlp`
- Feature normalization: `fit_zscore_per_feature`
- Target normalization: `fit_zscore_per_result`
- Warmup steps: `100`
- Updates per step: `1`
- Online LR: `1e-3`
- Validation fraction: `0.1`
- Heldout-test fraction: `0.2`
- Effective seed: `4`

Results:

| Feature mode | Hidden | Selected step | Validation min cosine | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `injection_grad_policy_state` | `16` | `100` | `0.19320` | `0.59421 / 0.39967` | `0.28987 / 0.46921` | `1.7669 / 1.1802` | cosine and gap fail |
| `injection_grad_policy_state` | `32` | `100` | `0.28199` | `0.43401 / 0.40230` | `0.53978 / 0.57451` | `1.8823 / 1.6198` | cosine and gap fail |
| `injection_grad_logits` | `16` | `30` | `0.27470` | `0.64364 / 0.47630` | `0.07113 / 0.18319` | `1.6959 / 1.3566` | cosine fail |
| `injection_grad_logits` | `32` | `80` | `0.31885` | `0.66913 / 0.70278` | `0.28301 / 0.26578` | `1.7016 / 1.4895` | result cosine and gap fail |

Feature-scale observations:

- Policy-state fit feature scale min/median/mean/max:
  `0.00000182 / 0.002204 / 0.1171 / 1.5177`.
- Logits-state fit feature scale min/median/mean/max:
  `0.001190 / 0.002598 / 0.2836 / 1.5177`.
- Feature z-scoring did not improve heldout model-gradient agreement and often
  worsened relative norms.

## Decision

```text
online_mlp_shadow_feedback_feature_standardization_negative
```

No Stage 1 early-lift run was launched.

## Next

Do not rerun plain fit-split feature z-scoring as novelty. Next work should
change the synthetic-gradient objective, add explicit regularization, or build
a more stable target construction.
