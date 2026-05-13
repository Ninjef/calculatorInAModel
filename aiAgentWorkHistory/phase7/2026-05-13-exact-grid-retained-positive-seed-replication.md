# Exact-Grid Retained-Positive Seed Replication

## Task

```text
aiAgentProjectTasks/2026-05-13-phase-7-seventh-task-Exact-grid-retained-positive-seed-replication.md
```

## Claim Tested

Does the exact-grid upstream-open result-boundary recipe that produced the
seed-2 retained positive replicate across additional effective seeds?

## Code Changes

None.

## Preflight

Commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
91 passed
```

The existing seed-2 full-grid parity artifact was present and still matched
the task baseline:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage0_full_grid_parity_gate/stage0_full_grid_parity_summary.json
```

Key values: `grid_examples=400`, `grid_duplicate_pairs=0`,
`hard_best_result_equals_true_sum=1.0`,
`tie_aware_true_result_best_fraction=1.0`, and
`semantic_decoder_delta=0.0`.

## Run Root

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_seed_replication
```

The task CLI seeds were `4` and `5`. The training script stores
`seed=args.seed + num_digits`, so the output directories are
`model-c-2digit-seed6` and `model-c-2digit-seed7`.

## Stage 1 Teaching Replication

Both seeds passed Stage 1.

| CLI seed | Selected checkpoint | Hard result acc | Full-enum learned-result best fraction | Mean learned-result minus best gap | Canonical normal / calc result | Injection-zero | Forced-random | Oracle-at-eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `stage1_seed_4/.../model-c-2digit-seed6/checkpoint_snapshots/step_00700_weights.pt` | `1.0000` | `1.0000` | `0.0000` | `1.0000 / 1.0000` | `0.0550` | `0.0225` | `1.0000` |
| `5` | `stage1_seed_5/.../model-c-2digit-seed7/checkpoint_snapshots/step_00750_weights.pt` | `0.9975` | `0.9975` | `0.0059` | `0.9975 / 0.9975` | `0.0550` | `0.0225` | `1.0000` |

Stage 1 movement:

| CLI seed | Group | L2 delta | Max abs | Changed tensors |
| --- | --- | ---: | ---: | ---: |
| `4` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `4` | `calculator_hook.result_proj` | `93.1714` | `6.0694` | `2/2` |
| `4` | upstream encoder | `4.8055` | `0.1961` | `14/29` |
| `5` | semantic decoder | `0.0` | `0.0` | `0/5` |
| `5` | `calculator_hook.result_proj` | `90.6737` | `5.4942` | `2/2` |
| `5` | upstream encoder | `5.1313` | `0.2050` | `14/29` |

## Stage 2 Target-Off Retention

Both seeds retained above the final `0.70` hard-result/full-enum floor, but
neither reached the stricter exact-grid best-post-start `90%` retention ratio.

| CLI seed | Stage 1 selected hard acc | Best post-start checkpoint | Best post-start exact-grid hard acc | Exact-grid retention ratio | Final exact-grid hard acc | Final learned-result best fraction | Final gap | Strict Stage 2 gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `4` | `1.0000` | `stage2_seed_4/.../model-c-2digit-seed6/checkpoint_snapshots/step_00100_weights.pt` | `0.8700` | `0.8700` | `0.8350` | `0.8350` | `0.8142` | fail |
| `5` | `0.9975` | `stage2_seed_5/.../model-c-2digit-seed7/checkpoint_snapshots/step_00150_weights.pt` | `0.8800` | `0.8822` | `0.7925` | `0.7925` | `0.9376` | fail |

Sampled canonical diagnostics for the selected Stage 2 checkpoints kept
injection-zero at `0.0550`, forced-random at `0.0225`, and oracle-at-eval at
`1.0000`. These are wiring/control checks only.

Stage 2 movement:

| CLI seed | Span | Semantic decoder L2 / max / changed | Result-proj L2 / max / changed | Upstream L2 / max / changed |
| --- | --- | ---: | ---: | ---: |
| `4` | start -> best step `100` | `0.0 / 0.0 / 0/5` | `1.3996 / 0.2410 / 2/2` | `0.0725 / 0.0081 / 14/29` |
| `4` | start -> final step `400` | `0.0 / 0.0 / 0/5` | `2.1418 / 0.2933 / 2/2` | `0.2140 / 0.0278 / 14/29` |
| `5` | start -> best step `150` | `0.0 / 0.0 / 0/5` | `1.7317 / 0.1574 / 2/2` | `0.1051 / 0.0117 / 14/29` |
| `5` | start -> final step `400` | `0.0 / 0.0 / 0/5` | `2.0321 / 0.3115 / 2/2` | `0.2185 / 0.0264 / 14/29` |

## Required Diagnostics

Canonical diagnostics were saved under per-step directories such as:

```text
checkpoint_snapshots/canonical_diagnostic_step_00700/
checkpoint_snapshots/canonical_diagnostic_step_00750/
checkpoint_snapshots/canonical_diagnostic_step_00100/
checkpoint_snapshots/canonical_diagnostic_step_00150/
checkpoint_snapshots/canonical_diagnostic_step_00400/
```

Full-enum diagnostics were saved under matching per-step directories:

```text
checkpoint_snapshots/full_enum_diagnostic_step_00700/
checkpoint_snapshots/full_enum_diagnostic_step_00750/
checkpoint_snapshots/full_enum_diagnostic_step_00100/
checkpoint_snapshots/full_enum_diagnostic_step_00150/
checkpoint_snapshots/full_enum_diagnostic_step_00400/
```

An additional exact-grid scan over all Stage 2 snapshots found the best
post-start checkpoints:

```text
CLI seed 4: step 00100, exact-grid hard result accuracy 0.8700
CLI seed 5: step 00150, exact-grid hard result accuracy 0.8800
```

## Interpretation

Label:

```text
exact_grid_seed_replication_negative
```

The teaching part of the recipe replicated: both seeds learned near-perfect
hard result requests with semantic decoder movement exactly `0.0`. Target-off
continuation also retained a materially useful result request above `0.70`
final exact-grid accuracy for both seeds.

However, under the task's strict Stage 2 pass gate, neither seed retained at
least `90%` of its selected Stage 1 hard result accuracy on the exact grid.
Therefore the seed-2 retained positive did not robustly replicate across CLI
seeds `4` and `5`.

## Recommendation

Do not proceed as if exact-grid retention has fully replicated. The next task
should analyze seed fragility and compare against multi-sample result-space
policy gradient with per-prompt or leave-one-out baselines. Do not return to
oracle/readout checks or frozen-head boundary-target variants.
