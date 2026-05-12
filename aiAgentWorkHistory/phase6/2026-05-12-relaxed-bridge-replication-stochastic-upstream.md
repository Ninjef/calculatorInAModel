# Relaxed Bridge Replication, Stochastic Gumbel, And Upstream-Open Stress

## Task

```text
aiAgentProjectTasks/2026-05-11-phase-6-seventh-task-Relaxed-bridge-replication-stochastic-and-upstream-open.md
```

## Code Added

- Added `scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py`.
- The runner executes and summarizes:
  - Stage 0 deterministic and stochastic Gumbel gradient gates;
  - Stage 1 deterministic Concrete replication across CLI seeds `0`, `2`, `3`
    / effective seeds `2`, `4`, `5`;
  - Stage 2 relaxation-off retention;
  - Stage 3 literal stochastic Gumbel plus the one allowed stabilization branch;
  - Stage 4 upstream-open deterministic stress and frozen/open retentions;
  - selected canonical, private, and full-enum diagnostics.

## Commands

All PyTorch run commands were executed outside the sandbox because sandboxed
subprocesses aborted with OpenMP shared-memory errors.

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py stage0 --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py stage1 --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py stage2 --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py stage3 --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py stage4 --jobs 1
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py diagnostics
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py summarize
```

Exact underlying `overfit_one_batch.py` and diagnostic commands are captured in
the per-branch log files under:

```text
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream
```

The Stage 2 run was interrupted once. The runner was updated to skip completed
branches on restart. One duplicate seed `4` retention artifact remains under
the run root, but summaries use the latest completed run per branch.

## Results

Run root:

```text
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream
```

Compact summary:

```text
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream/summary.md
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream/summary.json
```

### Stage 0

- Deterministic gate passed: best-pair probability delta `+2.0915e-05`,
  gradient cosine `0.2345`, input-proj delta `1.0896`, upstream delta `0.0`,
  semantic grad/delta `0.0 / 0.0`.
- Stochastic Gumbel gate passed across seeds `7201`, `7202`, `7203`: mean
  best-pair delta `+2.0271e-05`, all three gradient cosines positive, semantic
  grad/delta `0.0 / 0.0`.

### Stage 1 Deterministic Concrete

All three deterministic seeds crossed the fast gate:

| Effective seed | First gate | Best fast normal/operand/pair/calc | Final fast normal/operand/pair/calc |
| ---: | ---: | ---: | ---: |
| `2` | `200` | `1.000 / 1.000 / 1.000 / 1.000` | `0.859 / 0.859 / 0.859 / 0.859` |
| `4` | `250` | `0.961 / 0.961 / 0.961 / 0.961` | `0.844 / 0.844 / 0.844 / 0.844` |
| `5` | `275` | `0.977 / 0.977 / 0.977 / 0.977` | `0.922 / 0.922 / 0.922 / 0.922` |

Full diagnostics on selected Stage 1 checkpoints showed seed `2` cleanly exact
or near-exact, while seeds `4` and `5` were near-gated but below the strict
`>=0.98` diagnostic threshold.

### Stage 2 Relaxation-Off Retention

All three deterministic seeds retained or completed to exact fast-gate metrics
after the relaxation was off and all teacher/local/expected objectives were
inactive.

| Effective seed | Final fast normal/operand/pair/calc | Final eval | Full-enum learned-best |
| ---: | ---: | ---: | ---: |
| `2` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` | `1.000` |
| `4` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` | `0.9922` |
| `5` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` | `1.000` |

Final retained objective weights:

```text
answer_loss_weight=1.0
final_aux_operand_loss_weight=0.0
final_adaptive_interface_loss_weight=0.0
final_local_target_loss_weight=0.0
final_expected_answer_loss_weight=0.0
final_relaxed_calculator_entropy_weight=0.0
final_input_proj_anchor_weight=0.0
```

### Stage 3 Literal Stochastic Gumbel

Literal stochastic Gumbel training failed despite the positive gradient gate.

| Branch | Best fast normal/operand/pair/calc | Final fast normal/operand/pair/calc | Final eval |
| --- | ---: | ---: | ---: |
| primary | `0.000 / 0.023 / 0.023 / 0.055` | `0.000 / 0.008 / 0.008 / 0.008` | `0.000` |
| stabilized | `0.000 / 0.023 / 0.023 / 0.055` | `0.000 / 0.008 / 0.008 / 0.008` | `0.000` |

Both stochastic branches reached `NaN` losses after step `225`.

### Stage 4 Upstream-Open Stress

The upstream-open deterministic branch crossed the fast gate at step `225`.

| Metric | Value |
| --- | ---: |
| Best fast normal/operand/pair/calc | `0.961 / 0.961 / 0.961 / 0.961` |
| Final fast normal/operand/pair/calc | `0.664 / 0.664 / 0.664 / 0.664` |
| Input-proj delta to selected checkpoint | `24.6044` L2 |
| Upstream delta to selected checkpoint | `0.0400` L2 |
| Upstream tensors changed | `14 / 29` |
| Semantic decoder delta | `0.0` |

Relaxation-off retention from the upstream-open selected checkpoint succeeded
with both upstream frozen and upstream still open:

| Retention condition | Final fast normal/operand/pair/calc | Canonical operand/pair/calc | Full-enum gaps |
| --- | ---: | ---: | ---: |
| upstream frozen | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` |
| upstream open | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` |

## Interpretation Labels

```text
deterministic_concrete_positive
stochastic_gumbel_negative
upstream_open_positive
```

## Interpretation

Deterministic hard-forward / soft-backward Concrete answer-loss training now
replicates as a fast-gate learner across effective seeds `2`, `4`, and `5`.
The selected Stage 1 checkpoints for seeds `4` and `5` were not fully exact by
the strict diagnostic threshold, but relaxation-off answer-only retention
completed all three seeds to exact or near-exact protocols with all teacher,
local-target, expected-loss, and relaxed objectives inactive.

Literal stochastic Gumbel sampling is a clear negative under the tested primary
and stabilization settings: it stayed near chance and became numerically
unstable. The deterministic bridge also tolerated modest upstream movement:
upstream parameters changed measurably, semantic decoder movement stayed zero,
and the selected protocol survived relaxation-off retention with upstream
frozen and open.
