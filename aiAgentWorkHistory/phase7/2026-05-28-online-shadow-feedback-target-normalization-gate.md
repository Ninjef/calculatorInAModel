# 2026-05-28 Online Shadow-Feedback Target-Normalization Gate

## Summary

Added fit-split per-result z-score target normalization to the online MLP
shadow-feedback diagnostic. The MLP trains on normalized targets, but all
model-gradient diagnostics unnormalize predictions and compare raw induced
gradients with the boundary-target ceiling.

## Code

- Added `--shadow-feedback-target-normalization`.
- Added `fit_zscore_per_result` mode.
- Normalization stats are fit only on the fit split.
- Validation and heldout-test examples never update target normalization stats.
- Added tests for normalized diagnostic plumbing and fit-only normalizer stats.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
99 passed
```

## Experiment

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_target_norm_gate
```

Shared configuration:

- Target normalization: `fit_zscore_per_result`
- Warmup steps: `100`
- Updates per step: `1`
- Online LR: `1e-3`
- Validation fraction: `0.1`
- Heldout-test fraction: `0.2`

Target-normalization stats:

- Mean L2: `0.09148`
- Scale min / median / max: `0.0000617 / 0.15612 / 0.21795`
- Scale mean: `0.14742`
- Clamped scale count: `0`

Results:

| Hidden | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | --- |
| `64` | `0.71287 / 0.77383` | `0.22750 / 0.18105` | `1.1929 / 1.1339` | gap fail |
| `32` | `0.71194 / 0.75472` | `0.19129 / 0.18739` | `1.2276 / 1.2244` | gap fail |
| `16` | `0.72595 / 0.75493` | `0.17235 / 0.14584` | `1.4146 / 1.1848` | near miss |
| `8` | `0.55824 / 0.58359` | `0.19213 / 0.15603` | `1.5886 / 1.3430` | cosine fail |

## Decision

```text
online_mlp_shadow_feedback_target_normalization_partial_no_go
```

Target normalization improved the signal but did not clear the complete gate.
No Stage 1 run was launched.

## Next

Change shadow input/state or objective more substantially, such as richer
policy features, explicit regularization, a different loss, or a more stable
target construction.
