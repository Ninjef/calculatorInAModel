# Linear Shadow-Feedback Gate

## Task

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-twelfth-task-Linear-shadow-feedback-gate.md
```

## Claim Tested

Can a fit-once linear shadow-feedback module turn answer-loss gradients at the
calculator injection into useful result-space calculator-request updates?

## Code Changes

- Split shadow feedback into a fit step and a frozen apply step:
  `fit_linear_shadow_feedback_weights` and
  `fixed_linear_shadow_feedback_alignment_loss`.
- Added `--shadow-feedback-weight`, `--shadow-feedback-ridge`, and
  `--shadow-feedback-gradient-diagnostic-only`.
- Stage 1 with `--shadow-feedback-weight > 0` fits the linear map once before
  training, saves `shadow_feedback_weights.pt`, and does not recompute boundary
  targets inside the training loop.
- Added tests for the diagnostic path and CLI parsing.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
97 passed
```

## Stage 0: Linear Shadow Diagnostic

Run root:

```text
runs/2026-05-28_phase7_shadow_feedback_gradient_gate/2026-05-28_124655_804690_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Metrics:

| Metric | Value |
| --- | ---: |
| shadow result-proj grad L2 | `0.08958` |
| shadow upstream grad L2 | `0.03303` |
| shadow semantic decoder grad L2 | `0.0` |
| result-proj cosine vs boundary | `0.99834` |
| upstream cosine vs boundary | `0.98543` |
| linear feedback fit cosine | `0.46028` |

Decision:

```text
linear_shadow_feedback_stage0_alignment_pass
```

## Stage 1: Frozen Linear Shadow Early-Lift Smoke

Run root:

```text
runs/2026-05-28_phase7_shadow_feedback_gradient_gate/stage1_linear_shadow_feedback_early_lift/2026-05-28_124712_295240_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Setup:

- natural `0..19`, exhaustive `20 x 20` grid;
- `calculator_action_head=result_space`;
- `shadow_feedback_weight=1.0`;
- `shadow_feedback_ridge=0.001`;
- semantic decoder frozen;
- `answer_loss_weight=0.0`;
- `aux_operand_loss_weight=0.0`;
- `adaptive_interface_loss_weight=0.0`;
- `expected_answer_loss_weight=0.0`;
- `result_boundary_target_loss_weight=0.0`;
- no oracle actions;
- 200 steps with snapshots every 25.

Calibration:

| Metric | Value |
| --- | ---: |
| fit cosine | `0.46028` |
| target grad L2 | `0.04935` |
| predicted feedback L2 | `0.01522` |
| target best equals true sum | `1.0` |

Results:

| Metric | Value |
| --- | ---: |
| best snapshot normal exact / calc-result accuracy | `0.070` at step `75` |
| final exact match | `0.040` |
| final learned calc-result accuracy in training curve | `0.045` |
| best injection-zero exact match | `0.065` |
| oracle-at-eval exact match | `1.0` |

Decision:

```text
linear_shadow_feedback_stage0_alignment_pass_stage1_early_lift_negative
```

## Interpretation

- A fit-once linear shadow map can induce gradients almost perfectly aligned
  with the boundary-target ceiling at initialization.
- That local update agreement does not translate into useful discovery under a
  frozen map. The early Stage 1 smoke performed worse than the prior
  output-projection boundary-feedback run (`0.040` final exact vs `0.160`).
- Do not run a long continuation of this exact branch.
- Next work should test heldout-validated or online-trained shadow modules,
  and require early Stage 1 lift before long-run budget.
