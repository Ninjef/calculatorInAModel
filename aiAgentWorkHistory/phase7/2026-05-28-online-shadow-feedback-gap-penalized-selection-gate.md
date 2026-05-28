# 2026-05-28 Online Shadow-Feedback Gap-Penalized Selection Gate

## Summary

Added gap-penalized validation checkpoint selection to the online MLP
shadow-feedback diagnostic. This targets the specific failure left by
directional losses: good heldout direction with train-heldout gaps above the
gate.

## Code

- Added `--shadow-feedback-selection-score-mode`.
- Added `gap_penalized_min_cosine`.
- Added `--shadow-feedback-selection-gap-penalty`.
- Validation history now records train-validation result/upstream cosine gaps.
- Preserved the original validation min-cosine mode as the default.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
101 passed
```

## Experiment

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_gap_penalized_selection_gate
```

Shared configuration:

- Shadow mode: `online_mlp`
- Feature mode: `injection_grad_logits`
- Feature normalization: `none`
- Target normalization: `fit_zscore_per_result`
- Warmup steps: `100`
- Updates per step: `1`
- Online LR: `1e-3`
- Validation fraction: `0.1`
- Heldout-test fraction: `0.2`
- Effective seed: `4`

Results:

| Loss mode | Hidden | Gap penalty | Selected step | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `16` | `1.0` | `90` | `0.76465 / 0.80065` | `0.19855 / 0.15470` | `1.1896 / 1.0965` | gap fail |
| `cosine` | `16` | `3.0` | `80` | `0.74504 / 0.78002` | `0.18433 / 0.14696` | `1.2853 / 1.1359` | result-gap fail |
| `cosine` | `16` | `4.0` | `70` | `0.71652 / 0.74394` | `0.16727 / 0.13527` | `1.4029 / 1.1966` | result-gap fail |
| `cosine` | `16` | `5.0` | `60` | `0.68723 / 0.69794` | `0.15107 / 0.12203` | `1.5371 / 1.2740` | cosine fail |
| `cosine` | `32` | `1.0` | `90` | `0.79370 / 0.82697` | `0.20239 / 0.15449` | `1.0114 / 1.0177` | gap fail |
| `mse_plus_cosine` | `16` | `1.0` | `90` | `0.76966 / 0.80241` | `0.19924 / 0.15997` | `1.1803 / 1.0974` | gap fail |
| `mse_plus_cosine` | `32` | `1.0` | `90` | `0.78528 / 0.81739` | `0.20909 / 0.16364` | `1.0599 / 1.0650` | gap fail |

## Decision

```text
online_mlp_shadow_feedback_gap_penalized_selection_tradeoff_no_go
```

No Stage 1 early-lift run was launched.

## Next

Selection alone cannot cross heldout cosine and train-heldout gap gates
together. Next work should add training-time regularization, use a more stable
target construction, or change the learned-gradient state.
