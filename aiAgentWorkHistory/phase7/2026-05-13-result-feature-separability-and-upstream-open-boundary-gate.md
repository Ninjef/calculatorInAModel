# Result Feature Separability And Upstream-Open Boundary Gate

## Task

```text
aiAgentProjectTasks/2026-05-13-phase-7-fifth-task-Frozen-feature-result-separability-and-minimal-upstream-open-boundary-gate.md
```

## Claim Tested

Do the frozen strict natural `0..19` operand-span features contain enough
information for a result request head to recover the answer-derived result
target, or does Phase 7 need upstream representation movement?

## Code Changes

- Added `scripts/run_phase7_result_feature_separability.py`.
- Added `calculator_result_head_hidden_size` to `GPTConfig` and
  `scripts/overfit_one_batch.py`.
- `calculator_result_head_hidden_size=0` preserves the existing linear
  `calculator_hook.result_proj`.
- Positive hidden sizes create a one-hidden-layer result-space MLP while
  keeping the hard forward calculator path and `result_space` action head
  unchanged.
- Added focused tests for feature extraction shape, target parity after target
  construction, synthetic linear probe overfitting, probe CLI validation, and
  hidden result-head shape.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase7_result_feature_separability.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
88 passed
```

## Frozen Feature Separability

Command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase7_result_feature_separability.py
```

Artifacts:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_separability_summary.json
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_probe_all400.csv
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_probe_5fold.csv
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/result_feature_predictions.csv
```

Primary results:

| Metric | Value |
| --- | ---: |
| answer-derived target parity with true sum | `1.0000` |
| exact `result_proj` input width | `64` |
| linear all-400 accuracy | `0.9217` |
| linear 5-fold mean / min accuracy | `0.1358` / `0.0375` |
| MLP-64 all-400 / 5-fold mean accuracy | `1.0000` / `0.1400` |
| MLP-128 all-400 / 5-fold mean accuracy | `1.0000` / `0.1458` |
| operand-A span linear accuracy | `1.0000` |
| operand-B span linear accuracy | `1.0000` |

Interpretation:

- The exact frozen feature is not linearly sufficient by the task threshold.
- A shallow MLP can memorize the finite natural `0..19` grid exactly, so useful
  nonlinear information is present in the frozen features.
- Low held-out fold accuracy means this is an all-grid separability result, not
  evidence of broad smooth generalization.

Decision: proceed to the conditional small MLP result head.

## Conditional MLP Result Head

Run:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/stage1_mlp64_boundary_target/2026-05-13_091415_689135_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rhead64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `calculator_result_head_hidden_size=64`
- semantic decoder frozen
- upstream frozen
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `result_boundary_target_temperature=0.25`
- `input_proj_lr=0.01`
- `steps=600`

Result:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.2950` at step `600` |
| best learned-result best fraction | `0.2950` |
| mean learned-result minus best-result gap at best | `3.9422` |
| final eval exact | `0.2425` |

Decision: failed the `0.70` Stage 1 gate. Target-off retention was not run.

## Minimal Upstream-Open Boundary Target

Run:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open/stage1_upstream_open_boundary_target/2026-05-13_093849_217301_model-c-op0-19-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- linear result head, `calculator_result_head_hidden_size=0`
- semantic decoder frozen
- upstream open
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `result_boundary_target_temperature=0.25`
- `input_proj_lr=0.01`
- `upstream_lr=0.0003`
- `steps=600`

Best checkpoint:

```text
checkpoint_snapshots/step_00575_weights.pt
```

Stage 1 metrics:

| Metric | Value |
| --- | ---: |
| best hard learned calculator-result accuracy | `0.5975` at step `575` |
| best learned-result best fraction | `0.5975` |
| mean learned-result minus best-result gap at best | `2.0629` |
| final hard learned calculator-result accuracy | `0.4275` |
| final eval exact | `0.4625` |

Selected checkpoint diagnostics:

| Diagnostic | Value |
| --- | ---: |
| canonical normal exact | `0.5625` |
| canonical calculator-result accuracy | `0.5625` |
| canonical result-equivalent pair accuracy | `0.5625` |
| canonical pair exact | `0.0350` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| full-enum learned-result best fraction | `0.5900` |
| full-enum learned result matches true sum | `0.5900` |
| mean learned-result minus best-result gap | `2.0806` |
| true result best fraction | `1.0000` |
| tie-aware true best fraction | `1.0000` |
| soft target true result-group probability | `0.99994` |

Parameter movement from step `0` to selected step `575`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/3` |
| `calculator_hook.result_proj` | `42.2322` | `3.9242` | `2/2` |
| upstream encoder | `3.3516` | `0.1469` | `14/29` |

## Interpretation

Label:

```text
minimal_upstream_open_boundary_target_partial
```

The frozen features contain enough nonlinear information for all-grid shallow
result classification, but the production MLP result head did not pass Stage 1
teaching. Allowing upstream movement substantially improved the boundary-target
run, reaching `0.5975` hard learned result accuracy and `0.5900` full-enum
learned-result best fraction, while the semantic decoder stayed exactly fixed.

This is a partial rescue, not a pass. It did not reach the `0.70` gate and
drifted down by final, so target-off retention and stricter upstream-frozen
retention were not run.

## Recommendation

Do not run retention or seed replication from this checkpoint. Next work should
either improve the upstream-open boundary target with a genuinely different
stabilizing mechanism, or move to a different signal family such as
multi-sample policy gradient with per-prompt baselines, surrogate gradients, or
direct feedback alignment.
