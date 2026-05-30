# 2026-05-30 Left-Operand Routed Multi-Hook Support

## Context

Same-layer multi-hook support allowed multiple hooks to run, but every hook was
active on every example. The next many-calculator diagnostic needs a route so
we can measure active hook count, per-hook quality, and interference.

## Code

- Added `GPTConfig.calculator_hook_routing`.
- Supported modes:
  - `all`: existing behavior, every hook applies.
  - `left_operand_mod`: route each fixed-width prompt to one hook by final
    left-operand digit modulo `calculator_hook_count`.
- Added diagnostics:
  - `calculator_hook_route`
  - `calculator_hook_route_counts`
- Per-hook applied injections are now masked so non-routed examples receive
  zero injection from inactive hooks.
- Added `--calculator-hook-routing {all,left_operand_mod}` to
  `scripts/overfit_one_batch.py`.
- Config and metrics record `calculator_hook_routing`.

## Smoke

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --steps 0 --batch-size 4 --eval-samples 8 --run-root /tmp/codex_multi_hook_route_smoke --device cpu --calculator-hook-count 3 --calculator-hook-routing left_operand_mod --calculator-estimator ste
```

Observed:

- Run root included `model-c-hooks3-routeleft_operand_mod`.
- `config.json` had `calculator_hook_count=3`,
  `calculator_hook_routing=left_operand_mod`, and matching model fields.
- `metrics.json` had `calculator_hook_count=3`,
  `calculator_hook_routing=left_operand_mod`.

## Tests

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q
```

Result: `127 passed in 4.22s`.

## Interpretation

This is still implementation plumbing, not evidence that many calculators train
properly. It enables the next diagnostic: a small routed/task-partitioned run
that reports per-hook route balance, per-hook calculator-result accuracy,
candidate-scorer calls, and leakage/interference between hooks.
