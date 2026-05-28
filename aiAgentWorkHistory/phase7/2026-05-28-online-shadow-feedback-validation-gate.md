# 2026-05-28 Online Shadow-Feedback Validation Gate

## Summary

Added validation-selected checkpointing to the online MLP shadow-feedback
warmup diagnostic. The validation split is used only for selecting a shadow
checkpoint; the final gate is reported on a separate heldout test split.

## Code

- Added `--shadow-feedback-validation-fraction`.
- Added `--shadow-feedback-validation-every`.
- The diagnostic records validation history, selected step/update, selection
  score, selected train/validation/test metrics, final unselected metrics, and
  train-validation plus validation-test gaps.
- `online_mlp` remains diagnostic-only and still rejects
  `--shadow-feedback-weight > 0`.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
98 passed
```

## Experiment

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_validation_gate
```

Configuration:

- Hidden size: `64`
- Online LR: `1e-3`
- Warmup steps: `100`
- Updates per step: `1`
- Validation fraction: `0.1`
- Validation every: `10`
- Heldout test fraction: `0.2`

Result:

- Fit / validation / heldout-test batch: `280 / 40 / 80`
- Best validation checkpoint: step `60`, update `60`
- Best validation score: `0.47207`
- Selected train result/upstream cosine: `0.96499 / 0.96796`
- Selected validation result/upstream cosine: `0.48413 / 0.47207`
- Selected heldout-test result/upstream cosine: `0.64485 / 0.72658`
- Selected train-test result/upstream gap: `0.32013 / 0.24138`
- Validation-test result/upstream gap: `-0.16073 / -0.25451`
- Selected heldout-test relative norm: `1.3604 / 1.2857`
- Final unselected heldout-test result/upstream cosine: `0.69549 / 0.76165`

## Decision

```text
online_mlp_shadow_feedback_validation_selection_negative
```

Validation selection alone did not rescue the simple online MLP shadow module.
No Stage 1 run was launched.

## Next

Change the learned-gradient target or input state before another Stage 0B gate:
target normalization, stronger regularization, richer policy features, or a
different synthetic-gradient objective.
