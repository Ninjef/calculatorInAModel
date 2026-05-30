# Routed Multi-Hook Snapshot Metrics

## Context

The same-layer multi-hook path and `left_operand_mod` routing made
task-partitioned calculator diagnostics possible, but the standard snapshot
artifact still summarized the primary hook. That was enough for single-hook
runs, but it would make routed runs easy to misread: hook 1 examples could be
reported through hook 0's trace.

## Code Change

- `scripts/overfit_one_batch.py` now reads `diagnostics["calculator_hook_route"]`
  when building calculator trace rows.
- If a routed run provides `calculator_traces`, each row uses the active hook's
  trace rather than always using the primary hook trace.
- Snapshot rows now include routed aggregate fields:
  - `calculator_hook_route_distribution`
  - `calculator_hook_active_count`
  - `hook_{i}_route_count`
  - `hook_{i}_normal_exact_match`
  - `hook_{i}_operand_exact_match`
  - `hook_{i}_calculator_result_accuracy`
  - `hook_{i}_mean_sampled_logp`
- `tests/test_model.py` adds a routed snapshot test to ensure both routed hooks
  appear in the snapshot row and expose per-hook accuracy fields.

## Smoke

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 1 --steps 0 --batch-size 4 --eval-samples 8 --snapshot-every 1 --snapshot-samples 8 --run-root /tmp/codex_multi_hook_route_snapshot_smoke --device cpu --calculator-hook-count 2 --calculator-hook-routing left_operand_mod --calculator-estimator ste
```

The smoke wrote:

```text
/tmp/codex_multi_hook_route_snapshot_smoke/2026-05-30_132438_570268_model-c-op0-1-hooks2-routeleft_operand_mod/model-c-1digit-seed1/diagnostic_snapshots.csv
```

The CSV included:

```text
calculator_hook_route_distribution={"0": 4, "1": 4}
calculator_hook_active_count=2
hook_0_route_count=4
hook_1_route_count=4
hook_0_calculator_result_accuracy=0.0
hook_1_calculator_result_accuracy=0.0
```

The zero accuracies are expected for an untrained zero-step model; the important
result is that the routed columns are present and populated from the active
hook traces.

## Tests

```bash
python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py tests/test_assignment_scaling.py tests/test_research_memory.py -q
```

Results:

```text
128 passed in 5.27s
141 passed in 4.47s
```

## Interpretation

This is instrumentation, not a training result. It prevents the next routed
training diagnostic from confusing primary-hook quality with active-hook
quality, and it gives future agents route balance and per-hook policy metrics
in the standard snapshot artifact.

The next useful experiment is still the small task-partitioned training
diagnostic: exact/topk scorer-call accounting, route balance, per-hook
calculator-result accuracy, normal accuracy, and controls over training.
