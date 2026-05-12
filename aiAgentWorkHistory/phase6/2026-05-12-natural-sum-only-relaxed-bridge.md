# Natural Sum-Only Relaxed Bridge

## Task

```text
aiAgentProjectTasks/2026-05-12-phase-6-eighth-task-Natural-sum-only-relaxed-bridge.md
```

## Code Added

- Added `scripts/run_phase6_natural_sum_only_relaxed_bridge.py`.
- Extended `scripts/run_full_enum_action_loss_diagnostic.py` with result-aware
  full-enum metrics for the natural sum-only underidentified action space:
  learned-result best fraction, learned-result minus best-result NLL gap,
  best-result-group true-sum fraction, effective result counts, and same-sum
  near-best pair counts.
- Replaced result grouping with a backend-compatible per-result loop after MPS
  rejected `scatter_reduce`.

## Validation

```bash
python3 -m py_compile scripts/run_phase6_natural_sum_only_relaxed_bridge.py scripts/run_full_enum_action_loss_diagnostic.py
python3 -m pytest tests/test_model.py -q
```

Result:

```text
67 passed
```

## Commands

The first Stage 0 run was attempted in the sandbox and failed with the known
OpenMP shared-memory error:

```text
OMP: Error #179: Function Can't open SHM failed
```

The PyTorch experiment commands were then run outside the sandbox:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_natural_sum_only_relaxed_bridge.py stage0 --jobs 1
```

Fresh oracle wiring attempts:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 1000 --batch-size 64 --eval-samples 512 --operand-max 19 --calculator-operand-vocab-size 20 --oracle-train --calculator-estimator ste --answer-loss-weight 1.0 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-format sum --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --seed 0 --snapshot-every 250 --checkpoint-every 250 --snapshot-samples 128 --run-root runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle --log-every 50
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 3000 --batch-size 64 --eval-samples 512 --lr 0.001 --operand-max 19 --calculator-operand-vocab-size 20 --oracle-train --calculator-estimator ste --answer-loss-weight 1.0 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-format sum --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --seed 1 --snapshot-every 500 --checkpoint-every 500 --snapshot-samples 128 --run-root runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle_lr1e3 --log-every 100
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 2000 --batch-size 400 --eval-samples 512 --lr 0.003 --operand-max 19 --calculator-operand-vocab-size 20 --oracle-train --calculator-estimator ste --answer-loss-weight 1.0 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-format sum --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --seed 2 --snapshot-every 500 --checkpoint-every 500 --snapshot-samples 128 --run-root runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle_batch400 --log-every 100
```

## Results

Run root:

```text
runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge
```

Stage 0 using the existing strict sum-only oracle decoder:

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `0.9375` |
| Injection-zero exact | `0.0000` |
| Forced-random exact | `0.0547` |
| Initial hard answer exact | `0.0078` |
| Initial hard learned calculator-result accuracy | `0.0078` |
| Full-enum best result group matches true sum | `0.90625` |
| Full-enum true-pair best fraction | `0.078125` |
| Mean same-true-sum near-best pair count | `13.9297` |
| Mean effective action pairs | `30.1275` |
| Mean effective result count | `2.4661` |
| Semantic decoder delta | `0.0` |

Fresh oracle attempts:

| Branch | Eval exact | Diagnostic exact | Oracle-at-eval exact | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 steps, lr `0.003`, batch `64` | `0.9238` | `0.9453` | `0.9531` | `0.0078` | `0.0156` |
| 3000 steps, lr `0.001`, batch `64` | `0.9395` | `0.9297` | `0.9063` | `0.0000` | `0.0156` |
| 2000 steps, lr `0.003`, batch `400` | `0.9395` | `0.9141` | `0.9375` | `0.0078` | `0.0313` |

## Interpretation

The natural sum-only bridge stopped at Stage 0. The required wiring gate did
not pass with the existing checkpoint, and fresh oracle semantic decoder
attempts under the fixed tiny sum-only / operand-span / answer-decoder setup
also failed to reach `oracle-at-eval >= 0.98`.

This is a useful negative/blocker, not a relaxed-bridge training negative. The
downstream sum-only semantic decoder is not currently strong enough to make
learned-interface training interpretable.

Interpretation label:

```text
natural_sum_only_negative
sum_only_semantic_decoder_wiring_blocker
```

## Next Recommendation

Fix or strengthen the strict natural sum-only semantic decoder wiring first.
Only rerun deterministic Concrete answer-loss bridge training once Stage 0 has
oracle-at-eval and full-enum best-result-match near exact. Do not scale to
`operand_max=99` from this blocked state.
