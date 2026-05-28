# 2026-05-28 Online Shadow-Feedback Directional-Loss Gate

## Summary

Added directional loss modes to the online MLP shadow-feedback diagnostic.
This tests whether the MLP should optimize the same kind of direction signal
used by the model-gradient gate rather than only componentwise MSE.

## Code

- Added `--shadow-feedback-loss-mode`.
- Added `mse`, `cosine`, and `mse_plus_cosine` modes.
- Preserved `mse` as the default.
- Recorded loss mode and final fit objective in diagnostic summaries.
- Added tests for loss-mode behavior and CLI parsing/defaults.

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
runs/2026-05-28_phase7_online_shadow_feedback_directional_loss_gate
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

| Loss mode | Hidden | Selected step | Validation min cosine | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cosine` | `8` | `100` | `0.16149` | `0.59905 / 0.58589` | `0.17300 / 0.15366` | `1.5006 / 1.2359` | cosine fail |
| `cosine` | `16` | `90` | `0.70798` | `0.76465 / 0.80065` | `0.19855 / 0.15470` | `1.1896 / 1.0965` | gap fail |
| `cosine` | `32` | `90` | `0.52855` | `0.79370 / 0.82697` | `0.20239 / 0.15449` | `1.0114 / 1.0177` | gap fail |
| `mse_plus_cosine` | `8` | `100` | `0.15280` | `0.58189 / 0.58367` | `0.17915 / 0.15090` | `1.5508 / 1.2926` | cosine fail |
| `mse_plus_cosine` | `16` | `100` | `0.70406` | `0.77848 / 0.81119` | `0.20450 / 0.16076` | `1.1373 / 1.0900` | gap fail |
| `mse_plus_cosine` | `32` | `90` | `0.53209` | `0.78528 / 0.81739` | `0.20909 / 0.16364` | `1.0599 / 1.0650` | gap fail |

## Decision

```text
online_mlp_shadow_feedback_directional_loss_partial_no_go
```

No Stage 1 early-lift run was launched.

## Next

Directional loss improved heldout direction and relative norms, especially for
`cosine` h32, but did not solve train-heldout gap. Next work should add
explicit norm/gap regularization, use a more stable target construction, or
change the learned-gradient state more substantially.
