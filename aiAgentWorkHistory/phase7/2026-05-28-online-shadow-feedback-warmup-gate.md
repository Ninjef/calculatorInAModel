# 2026-05-28 Online Shadow-Feedback Warmup Gate

## Summary

Implemented an online MLP shadow-feedback warmup diagnostic for Phase 7. The
diagnostic trains a local shadow module, not `TinyGPT`, and then checks whether
the module-induced model gradients agree with the boundary-target ceiling on a
heldout split.

## Code

- Added `ShadowFeedbackMLP` and online shadow-feedback helpers in
  `scripts/overfit_one_batch.py`.
- Added `--shadow-feedback-mode {fit_once_linear,online_mlp}` plus online
  warmup flags.
- Kept `online_mlp` diagnostic-only for now; using it with
  `--shadow-feedback-weight > 0` is rejected.
- Added tests covering heldout model-gradient reporting, unchanged main-model
  parameters during warmup, and CLI parsing.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
98 passed
```

## Experiments

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_gate
```

Hidden size `64`, `lr=1e-3`, `100` warmup steps:

- Fit/heldout split: `320 / 80`
- Fit MSE / prediction cosine: `0.01747 / 0.55395`
- Train result/upstream cosine: `0.98498 / 0.98030`
- Heldout result/upstream cosine: `0.71673 / 0.76010`
- Train-heldout result/upstream gap: `0.26825 / 0.22020`
- Heldout result/upstream relative norm: `1.2678 / 1.2188`
- Heldout semantic decoder grad L2: `0.0`

Hidden size `16`, `lr=1e-3`, `100` warmup steps:

- Fit MSE / prediction cosine: `0.02372 / 0.24175`
- Train result/upstream cosine: `0.61852 / 0.50238`
- Heldout result/upstream cosine: `0.62555 / 0.66675`
- Train-heldout result/upstream gap: `-0.00703 / -0.16437`
- Heldout result/upstream relative norm: `1.5462 / 1.1396`

## Decision

```text
online_mlp_shadow_feedback_stage0b_partial_alignment_no_clean_gate
```

The online MLP direction is better than fit-once linear shadow feedback, but
this simple form did not clear the full gate. Hidden size `64` had adequate
heldout cosines but too much train-heldout gap. Hidden size `16` reduced the
gap but missed the result-head alignment threshold. No Stage 1 run was
launched.

## Next

Try a genuinely stronger shadow-generalization mechanism: validation early
stopping, regularization, target normalization, richer result-policy state, or
a different synthetic-gradient objective. Gate heldout before Stage 1.
