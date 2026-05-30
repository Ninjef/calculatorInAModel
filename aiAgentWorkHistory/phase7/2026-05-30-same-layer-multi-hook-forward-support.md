# 2026-05-30 Same-Layer Multi-Hook Forward Support

## Context

The previous many-calculator accounting review showed that policy-topk reduces
per-calculator candidate scoring but does not prove many-calculator scaling.
The repo still needed a concrete path to instantiate more than one calculator
hook before a real routed/multi-hook diagnostic could be run.

## Code

- Added `GPTConfig.calculator_hook_count`.
- Kept the first hook under the existing `calculator_hook.*` state-dict names
  for checkpoint compatibility.
- Added extra independent hooks under `extra_calculator_hooks.*`.
- Combined same-layer hook injections by summing them once at the read site.
- Added diagnostics:
  - `calculator_active_hook_count`
  - `calculator_hook_injections`
  - `calculator_traces`
- Added `--calculator-hook-count` to `scripts/overfit_one_batch.py`.
- Updated freezing and adaptive optimizer grouping so extra hook input/pair/result
  heads are handled with the primary hook instead of silently falling into
  `upstream`.

## Smoke

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --steps 0 --batch-size 4 --eval-samples 8 --run-root /tmp/codex_multi_hook_smoke --device cpu --calculator-hook-count 3 --calculator-estimator ste
```

Observed:

- Run root included `model-c-hooks3`.
- `config.json` had `calculator_hook_count=3` and
  `model.calculator_hook_count=3`.
- `metrics.json` had `calculator_hook_count=3`.
- Trainable parameter groups were `calculator_hook.input_proj` and `upstream`.

## Tests

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py tests/test_assignment_scaling.py -q
```

Result: `128 passed in 5.59s`.

## Interpretation

This is a prerequisite implementation result, not evidence that the training
goal is solved. The model can now host multiple same-layer independent hooks
and report their activity, but it still needs routing/task partitioning and a
diagnostic that measures per-hook policy quality, scorer calls, and
interference/leakage.
