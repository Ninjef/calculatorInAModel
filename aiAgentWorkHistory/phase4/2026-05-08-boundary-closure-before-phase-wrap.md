# 2026-05-08 - Boundary closure before phase wrap

Task: close the loose Phase 4 partial-handoff boundary before deciding whether
to wrap the phase.

## Claim

Answer loss can complete and stabilize a partially taught calculator-query
protocol after direct operand supervision is removed, but the handoff boundary
should have nearby failed lower neighbors for the remaining loose seeds before
Phase 4 is wrapped.

## Runner

Added:

```text
scripts/run_phase4_boundary_closure.py
```

Run root:

```text
runs/2026-05-08_phase4_boundary_closure
```

The runner reuses the existing dense Stage 1A snapshots from:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage1a
```

Stage 1A was not rerun.

## Shared setup

- Stage 0B semantic decoder:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=adaptive_interface`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- Stage 2 handoffs used `answer_loss_weight=1.0`,
  `adaptive_interface_loss_weight=0.0`, `aux_operand_loss_weight=0.0`,
  `input_proj_anchor_weight=0.0`
- Trainable parameters: `calculator_hook.input_proj` only (`1320` params)

## Selected handoffs

| Effective seed | CLI seed | Stage 1 step | Stage 1 operand exact | Reason |
| ---: | ---: | ---: | ---: | --- |
| `2` | `0` | `30` | `0.395` | lower midpoint between failed step `25` and retained step `60` |
| `2` | `0` | `55` | `0.438` | upper midpoint immediately before retained step `60` |
| `5` | `3` | `20` | `0.027` | very-low below-boundary probe |
| `5` | `3` | `25` | `0.078` | nearest lower neighbor below retained step `30` |

Seed `4` was skipped because the previous task already bracketed it at adjacent
steps `30` failed and `35` retained.

## Fast gates

| Effective seed | Handoff | Run path | Final operand/pair/calc | Injection-zero | Forced-random | Oracle | Status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `2` | step `30` | `runs/2026-05-08_phase4_boundary_closure/stage2/seed2/step30/2026-05-08_072232_382511_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `0.809` / `0.809` / `0.809` | `0.000` | `0.016` | `1.000` | failed |
| `2` | step `55` | `runs/2026-05-08_phase4_boundary_closure/stage2/seed2/step55/2026-05-08_072232_382505_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `0.844` / `0.844` / `0.844` | `0.000` | `0.016` | `1.000` | failed, nearest below step `60` |
| `5` | step `20` | `runs/2026-05-08_phase4_boundary_closure/stage2/seed5/step20/2026-05-08_072232_382502_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `0.734` / `0.734` / `0.734` | `0.004` | `0.020` | `1.000` | failed |
| `5` | step `25` | `runs/2026-05-08_phase4_boundary_closure/stage2/seed5/step25/2026-05-08_072232_382502_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `0.855` / `0.855` / `0.855` | `0.004` | `0.020` | `1.000` | failed, nearest below step `30` |

All four final checkpoints had:

- `final_aux_operand_loss_weight=0.0`
- `final_adaptive_interface_loss_weight=0.0`
- `final_input_proj_anchor_weight=0.0`
- frozen semantic decoder and frozen upstream encoder
- trainable parameter groups limited to `calculator_hook.input_proj`

## Diagnostics

Diagnostics were run on all four closure handoffs because each one was selected
specifically to change the boundary conclusion.

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-best | Full-enum learned-true/best gaps |
| --- | ---: | ---: | ---: | ---: |
| seed `2`, step `30` | `0.848` / `0.848` / `0.848` | `0.828` / `0.828` / `0.828` | `0.758` | `0.839` / `0.839` |
| seed `2`, step `55` | `0.855` / `0.855` / `0.855` | `0.845` / `0.845` / `0.845` | `0.852` | `0.705` / `0.705` |
| seed `5`, step `20` | `0.727` / `0.727` / `0.727` | `0.723` / `0.723` / `0.723` | `0.711` | `0.995` / `0.995` |
| seed `5`, step `25` | `0.828` / `0.828` / `0.828` | `0.848` / `0.848` / `0.850` | `0.883` | `0.349` / `0.349` |

The diagnostics agree with the fast gates: these are partial learned protocols,
not retained true-operand calculator-query protocols.

## Closed boundary

| Effective seed | Nearest failed lower handoff | Lowest retained handoff | Status |
| ---: | --- | --- | --- |
| `2` | step `55`, Stage 1 operand `0.438`, final `0.844` | step `60`, Stage 1 operand `0.641`, final `1.000` | narrowed |
| `4` | step `30`, Stage 1 operand about `0.19`, final `0.699` | step `35`, Stage 1 operand `0.363`, final `0.980` | already bracketed |
| `5` | step `25`, Stage 1 operand `0.078`, final `0.855` | step `30`, Stage 1 operand about `0.20`, final `0.980` | lower failed neighbor established |

## Interpretation

Seed `2` now has a much tighter lower failed neighbor: the boundary is no
longer `0.320` failed vs `0.641` retained, but `0.438` failed vs `0.641`
retained.

Seed `5` remains unusually permissive, but it is not unbounded. The step `25`
handoff recovered substantially under answer loss, but stayed below retention
on fast gates, private all-pair decoding, and full-enum action-loss diagnostics.

No new retained checkpoint exposed a confound. The new runs are all below the
retention threshold and have positive full-enum gaps, so they support failed
lower-neighbor claims rather than answer-only shortcut claims.

## Recommendation

Wrap Phase 4.

The phase result is now sharp enough:

```text
With an identifiable answer target and a frozen readable upstream
representation, answer loss can complete a partially taught calculator-query
protocol after direct operand supervision is exactly removed, but only above a
seed-dependent handoff quality.
```

Do not spend another Phase 4 task on this boundary unless a future reader wants
finer seed-specific interpolation. The next phase should choose a larger step:
upstream discovery, transfer, or a broader identifiable task.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase4_boundary_closure.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase4_boundary_closure.py stage2 --jobs 4
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase4_boundary_closure.py summarize
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase4_boundary_closure.py diagnostics
```
