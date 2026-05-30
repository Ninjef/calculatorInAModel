# 2026-05-30 - Shared routed output projection

Task: remove the cloned per-hook output-projection parameter growth left after
active-only routed execution.

## Question

Can routed calculator hooks share one result-to-residual semantic output
projection instead of cloning one projection per hook?

The four-hook routed source/handoff result needed cloned output projections so
each hook had the same semantic interface. That made the training result fair,
but parameter count still grew linearly with hook count.

## Changes

- Added `GPTConfig.calculator_share_output_proj`.
- Added `TinyGPT.tie_calculator_output_projections()`.
- Added `--share-calculator-output-proj` to `scripts/overfit_one_batch.py`.
- Propagated the flag through `make_model_config`, additive handoff probe
  construction, config JSON, and metrics JSON.
- Made `--share-calculator-output-proj` mutually exclusive with
  `--clone-primary-calculator-output-proj`.
- Canonicalized tied-model state dict loading so older untied checkpoints use
  the primary hook's output projection value rather than an extra hook's old
  projection.
- Updated semantic-decoder parameter grouping so extra-hook output projections
  are treated as semantic decoder parameters.

## Validation

Focused tests:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py::test_shared_calculator_output_projection_ties_extra_hooks tests/test_model.py::test_shared_calculator_output_projection_loads_primary_checkpoint_value tests/test_model.py::test_share_primary_calculator_output_projection_reduces_parameters tests/test_model.py::test_clone_primary_calculator_output_projection_to_extra_hooks tests/test_model.py::test_freeze_semantic_decoder_preserves_decoder_but_not_interface -q
```

Result: `5 passed`.

Broader regression set:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py tests/test_assignment_scaling.py tests/test_research_memory.py -q
```

Result: `149 passed`.

CLI smoke:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 2 --exhaustive-grid-batch --calculator-operand-vocab-size 3 --steps 0 --batch-size 9 --eval-samples 9 --snapshot-every 1 --snapshot-samples 9 --n-layer 1 --n-head 1 --n-embd 8 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-hook-count 3 --calculator-hook-routing left_operand_mod --share-calculator-output-proj --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 1 --calculator-bottleneck-mode none --calculator-output-format sum --run-root /tmp/codex_shared_output_proj_smoke --device cpu
```

The smoke wrote `share_calculator_output_proj=True` in both `config.json` and
`metrics.json`.

## Result

This is a positive implementation result for the many-calculator parameter
axis. A routed model can now use many hook-specific policies while sharing one
semantic output projection.

## Interpretation

This removes the need for cloned per-hook output projections as a fair routed
semantic interface. It does not yet prove tied-output routed training, because
the known source/handoff positives used cloned projections. The next empirical
gate should run the routed source recipe with `--share-calculator-output-proj`
and compare per-hook calculator accuracy, injection-zero controls, and handoff
quality against the cloned-output baseline.
