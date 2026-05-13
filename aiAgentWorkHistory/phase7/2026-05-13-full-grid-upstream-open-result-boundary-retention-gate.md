# Full-Grid Upstream-Open Result Boundary Retention Gate

## Task

```text
aiAgentProjectTasks/2026-05-13-phase-7-sixth-task-Full-grid-upstream-open-result-boundary-retention-gate.md
```

## Claim Tested

Can the answer-derived result-boundary target teach the natural `0..19`
result-space interface when every ordered pair is present on every training
step, and does that hard result request survive after the boundary target is
removed?

## Code Changes

- Added `--exhaustive-grid-batch` to `scripts/overfit_one_batch.py`.
- Added `make_exhaustive_range_batch(...)`, which builds every ordered pair in
  `0..operand_max x 0..operand_max` exactly once using the same padding and
  loss-mask path as `make_range_batch`.
- Reused the fixed exhaustive batch on every training step when the flag is
  enabled.
- Recorded `exhaustive_grid_batch` and `exhaustive_grid_size` in both
  `config.json` and `metrics.json`.
- Added tests for exact ordered-pair coverage, padding/mask parity,
  validation of `--exhaustive-grid-batch` without `--operand-max`, and
  full-grid result-boundary gradient flow with semantic decoder movement held
  at zero.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
91 passed
```

## Stage 0 Full-Grid Parity Gate

Artifact:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage0_full_grid_parity_gate/stage0_full_grid_parity_summary.json
```

| Metric | Value |
| --- | ---: |
| grid examples | `400` |
| duplicate ordered pairs | `0` |
| hard-best result equals true sum | `1.0000` |
| tie-aware true-result best fraction | `1.0000` |
| soft target true-result probability | `0.99989` |
| target entropy | `0.00105` |
| effective result count | `1.00105` |
| initial hard learned result accuracy | `0.0225` |
| result-proj gradient L2 | `0.08966` |
| upstream gradient L2 | `0.03320` |
| semantic decoder gradient/delta L2 | `0.0 / 0.0` |

Gate passed.

## Stage 1 Exact-Grid Upstream-Open Teaching

Run:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- `exhaustive_grid_batch=true`, grid size `400`
- `answer_loss_weight=0.0`
- `result_boundary_target_loss_weight=1.0`
- `result_boundary_target_mode=hard_best_result`
- `result_boundary_target_temperature=0.25`
- `calculator_result_head_hidden_size=0`
- semantic decoder frozen, upstream open
- `input_proj_lr=0.01`, `upstream_lr=0.0003`
- `steps=800`, `snapshot_every=25`, `checkpoint_every=25`

Selected checkpoint:

```text
checkpoint_snapshots/step_00800_weights.pt
```

Selection reason: best hard learned calculator-result accuracy in the dense
curve.

| Metric | Value |
| --- | ---: |
| hard learned calculator-result accuracy | `0.9675` |
| full-enum learned-result best fraction | `0.9675` |
| full-enum learned result matches true sum | `0.9675` |
| mean learned-result minus best-result gap | `0.1108` |
| canonical normal exact | `0.9600` |
| canonical calculator-result accuracy | `0.9600` |
| canonical result-equivalent pair accuracy | `0.9600` |
| canonical pair exact | `0.0825` |
| injection-zero exact | `0.0550` |
| forced-random exact | `0.0225` |
| oracle-at-eval exact | `1.0000` |
| final eval exact | `0.9530` |

Parameter movement from Stage 1 step `0` to step `800`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| `calculator_hook.result_proj` | `81.5030` | `4.3182` | `2/2` |
| upstream encoder | `4.6336` | `0.1954` | `14/29` |
| other interface groups | `0.0` | `0.0` | `0/0` |

Stage 1 passed, so the planned MLP rescue was skipped.

## Stage 2 Target-Off Retention

Run:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage2_target_off_full_grid/2026-05-13_154541_041524_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Setup:

- initialized from Stage 1 step `800`
- `answer_loss_weight=1.0`
- `result_boundary_target_loss_weight=0.0`
- aux/adaptive/expected/relaxed-entropy/anchor objectives all `0.0`
- semantic decoder frozen, upstream open
- `exhaustive_grid_batch=true`
- `steps=400`, `snapshot_every=25`, `checkpoint_every=25`

Best post-start target-off checkpoint:

```text
checkpoint_snapshots/step_00375_weights.pt
```

Final checkpoint:

```text
checkpoint_snapshots/step_00400_weights.pt
```

| Metric | Best post-start | Final |
| --- | ---: | ---: |
| hard learned calculator-result accuracy | `0.8800` | `0.8325` |
| full-enum learned-result best fraction | `0.8800` | `0.8325` |
| full-enum learned result matches true sum | `0.8800` | `0.8325` |
| canonical normal exact | `0.8775` | `0.8275` |
| canonical calculator-result accuracy | `0.8775` | `0.8275` |
| injection-zero exact | `0.0550` | `0.0550` |
| forced-random exact | `0.0225` | `0.0225` |
| oracle-at-eval exact | `1.0000` | `1.0000` |

Best post-start retention ratio:

```text
0.8800 / 0.9675 = 0.9096
```

Parameter movement from Stage 2 start to final step `400`:

| Group | L2 delta | Max abs | Changed tensors |
| --- | ---: | ---: | ---: |
| semantic decoder | `0.0` | `0.0` | `0/5` |
| `calculator_hook.result_proj` | `2.4398` | `0.2809` | `2/2` |
| upstream encoder | `0.2393` | `0.0372` | `14/29` |
| other interface groups | `0.0` | `0.0` | `0/0` |

## Interpretation

Label:

```text
full_grid_upstream_open_result_boundary_retained_positive
```

Exact ordered-grid coverage stabilized the upstream-open result-boundary
branch. The boundary target taught a hard result-space calculator request to
`0.9675` exact-grid hard result accuracy, and after the target was removed the
request retained above the required floor: final hard/full-enum result accuracy
remained above `0.70`, and the best post-start target-off checkpoint retained
more than `90%` of the selected Stage 1 hard result accuracy.

Oracle-at-eval, injection-zero, and forced-random are regression controls only.
The substantive Phase 7 result is retained learned hard calculator-result
behavior with semantic decoder movement exactly `0.0` and direct/auxiliary
operand supervision exactly off.

## Recommendation

Replicate this exact-grid retained positive across additional seeds before
claiming robustness. If replication holds, move to canonical-query/protocol
stabilization; if it fails, compare this branch against multi-sample
result-space policy gradient with per-prompt or leave-one-out baselines.
