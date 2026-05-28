# 2026-05-28 Online Shadow-Feedback Policy-State Feature Gate

## Summary

Added a richer policy-state input mode to the online MLP shadow-feedback
diagnostic and tested it under the target-normalized, validation-selected
Stage 0B gate. The new state did not improve generalization enough for Stage 1.

## Code

- Added `--shadow-feedback-feature-mode`.
- Preserved `injection_grad_logits` as the default feature mode.
- Added `injection_grad_policy_state`.
- Added mode-dependent shadow MLP input dimensions.
- Recorded feature mode, dimension, total feature norm, and per-block norms for
  input-gradient, logits, probabilities, log-probabilities, and entropy.
- Added tests for policy-state diagnostic plumbing and CLI parsing defaults.

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
runs/2026-05-28_phase7_online_shadow_feedback_policy_state_gate
```

Shared configuration:

- Shadow mode: `online_mlp`
- Feature mode: `injection_grad_policy_state`
- Target normalization: `fit_zscore_per_result`
- Warmup steps: `100`
- Updates per step: `1`
- Online LR: `1e-3`
- Validation fraction: `0.1`
- Heldout-test fraction: `0.2`

Results:

| Hidden | Selected step | Validation min cosine | Heldout result/upstream cosine | Train-heldout gap | Heldout relative norm | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `16` | `70` | `0.50705` | `0.68622 / 0.73909` | `0.23942 / 0.20316` | `1.1162 / 0.8952` | result cosine and gap fail |
| `32` | `90` | `0.59621` | `0.70372 / 0.76105` | `0.28529 / 0.21305` | `1.3519 / 1.2201` | gap fail |

Feature observations:

- Feature dimension was `134`.
- The log-probability block dominated raw feature scale:
  fit feature L2 `393.90`, input-gradient L2 `69.50`, and log-probability L2
  `382.84`.
- Final heldout cosines could look passable, but selected-checkpoint gaps were
  too wide for a clean Stage 1 go signal.

## Decision

```text
online_mlp_shadow_feedback_policy_state_raw_features_negative
```

No Stage 1 early-lift run was launched.

## Next

Do not rerun raw appended policy-state features as novelty. Next work should
try feature scaling/standardization, explicit regularization, a different
synthetic-gradient loss, or a more stable target construction.
