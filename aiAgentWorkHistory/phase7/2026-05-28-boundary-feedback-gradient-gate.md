# Boundary-Feedback Result-Space Gradient Gate

## Task

```text
aiAgentProjectTasks/completed/phase7/2026-05-28-phase-7-eleventh-task-Boundary-feedback-result-space-gradient-gate.md
```

## Claim Tested

Can an explicitly biased backward channel from answer loss at the calculator
boundary train natural `0..19` result-space calculator requests?

## Code Changes

- Added `calculator_estimator=direct_feedback_alignment` for result-space
  calculator actions.
- Added `boundary_feedback_alignment_loss`, which computes answer-loss
  gradients at the calculator injection under the frozen answer decoder, maps
  them into result-logit feedback, and trains result logits with a detached
  surrogate gradient.
- Added `--boundary-feedback-*` CLI flags and
  `--boundary-feedback-gradient-diagnostic-only`.
- Added tests for result-proj/upstream gradient flow, diagnostic summaries,
  and CLI parsing.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
96 passed
```

## Stage 0: Output-Projection Feedback

Run root:

```text
runs/2026-05-28_phase7_boundary_feedback_gradient_gate/2026-05-28_123023_283719_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Summary artifact:

```text
boundary_feedback_gradient_diagnostic_summary.json
```

Metrics:

| Metric | Value |
| --- | ---: |
| feedback result-proj grad L2 | `0.01867` |
| feedback upstream grad L2 | `0.00544` |
| feedback semantic decoder grad L2 | `0.0` |
| boundary result-proj grad L2 | `0.08969` |
| boundary upstream grad L2 | `0.03322` |
| result-proj cosine vs boundary | `0.27723` |
| upstream cosine vs boundary | `0.43823` |

Decision:

```text
boundary_feedback_output_projection_stage0_alignment_pass
```

## Stage 1: Output-Projection Feedback Discovery

Run root:

```text
runs/2026-05-28_phase7_boundary_feedback_gradient_gate/stage1_output_proj_feedback_discovery/2026-05-28_123042_173625_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Setup:

- natural `0..19`, exhaustive `20 x 20` grid;
- `calculator_action_head=result_space`;
- `boundary_feedback_weight=1.0`;
- semantic decoder frozen;
- `answer_loss_weight=0.0`;
- `aux_operand_loss_weight=0.0`;
- `adaptive_interface_loss_weight=0.0`;
- `expected_answer_loss_weight=0.0`;
- `result_boundary_target_loss_weight=0.0`;
- no oracle actions.

Results:

| Metric | Value |
| --- | ---: |
| best snapshot normal exact / calc-result accuracy | `0.155` at step `800` |
| final exact match | `0.160` |
| final learned calc-result accuracy in training curve | `0.150` |
| final injection-zero exact match | `0.0625` |
| final forced-random exact match | `0.0156` |
| final oracle-at-eval exact match | `1.0` |

Decision:

```text
boundary_feedback_stage1_discovery_negative
```

The aligned local feedback signal was not sufficient to discover the natural
hard result request. It improved above random-ish baselines but stayed far below
the `0.70` Stage 1 discovery floor, so no retention run was launched.

## Stage 0: Fixed-Random Direct Feedback

Run root:

```text
runs/2026-05-28_phase7_boundary_feedback_gradient_gate/2026-05-28_123623_484581_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4
```

Metrics:

| Metric | Value |
| --- | ---: |
| feedback result-proj grad L2 | `0.00378` |
| feedback upstream grad L2 | `0.00117` |
| feedback semantic decoder grad L2 | `0.0` |
| result-proj cosine vs boundary | `-0.00363` |
| upstream cosine vs boundary | `0.45997` |

Decision:

```text
fixed_random_direct_feedback_stage0_result_head_alignment_negative
```

No long fixed-random DFA training was run because the result-proj cosine failed
the Stage 0 gate.

## Interpretation

- Explicit boundary feedback can produce a locally aligned result-space update
  when the feedback matrix is the frozen calculator output projection.
- Local alignment alone is still insufficient: the Stage 1 run plateaued at
  `0.16` final exact match.
- A single fixed-random DFA matrix did not pass the result-head Stage 0 gate.
- Next work should use a learned shadow-gradient/synthetic-gradient module or a
  stronger feedback objective, with the same exact-grid Stage 0 gate and an
  early Stage 1 lift check before long-run budget.
