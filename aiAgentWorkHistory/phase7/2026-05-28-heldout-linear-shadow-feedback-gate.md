# Heldout Linear Shadow-Feedback Gate

## Task

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-thirteenth-task-Heldout-linear-shadow-feedback-gate.md
```

## Claim Tested

Does fit-once linear shadow-feedback alignment generalize off the calibration
examples, or was the previous same-batch Stage 0 pass over-optimistic?

## Code Changes

- Added `--shadow-feedback-heldout-fraction`.
- Extended `run_shadow_feedback_gradient_diagnostic` to fit on a deterministic
  split and report fit/train/heldout metrics separately.
- Added a fixed-weight shadow diagnostic helper for evaluating a fitted shadow
  map on arbitrary batches.
- Updated tests for heldout diagnostic plumbing and CLI parsing.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
97 passed
```

## Stage 0: Heldout Linear Shadow Diagnostic

Run root:

```text
runs/2026-05-28_phase7_shadow_feedback_heldout_gate/2026-05-28_125706_830132_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Setup:

- natural `0..19`, exhaustive `20 x 20` grid;
- deterministic `320/80` fit/heldout split (`--shadow-feedback-heldout-fraction 0.2`);
- `calculator_action_head=result_space`;
- semantic decoder frozen;
- exact boundary-target ceiling used only to fit/evaluate the diagnostic map;
- no Stage 1 training.

Metrics:

| Metric | Value |
| --- | ---: |
| fit batch / heldout batch | `320 / 80` |
| fit linear feedback cosine | `0.46857` |
| train result-proj cosine vs boundary | `0.99813` |
| heldout result-proj cosine vs boundary | `0.26221` |
| train upstream cosine vs boundary | `0.98449` |
| heldout upstream cosine vs boundary | `0.51012` |
| train-heldout result-proj cosine gap | `0.73591` |
| train-heldout upstream cosine gap | `0.47437` |
| heldout result/upstream relative norm | `1.1164 / 1.0291` |
| heldout semantic decoder grad L2 | `0.0` |

Decision:

```text
heldout_linear_shadow_feedback_stage0_generalization_negative
```

## Interpretation

- Same-batch model-update alignment was not a reliable gate for this linear
  shadow map. It overfit the calibration split at the result-head level.
- The heldout result-proj cosine `0.26221` is below the proposed `0.45` go
  threshold for online shadow warmup, and the `0.73591` train-heldout result
  gap is far above the proposed `0.35` stop threshold.
- Do not run more fit-once linear shadow training variants from this setup.
- Next work should add an online MLP shadow-feedback module that includes
  result-policy state and must pass heldout warmup before any Stage 1 early
  lift smoke.
