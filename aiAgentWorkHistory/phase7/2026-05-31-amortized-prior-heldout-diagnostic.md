# 2026-05-31 - Amortized Prior Heldout Diagnostic

## Question

Can a learned prior trained only from discovered prompt-memory targets supply
useful calculator-result targets for prompts that were not forced-scored?

## Implementation

- Added `OperandResultPrior`, an operand-conditioned target prior trained from
  prompt hard-memory entries.
- Added online training/replay flags:
  `--result-boundary-target-amortized-prior-weight`,
  `--result-boundary-target-amortized-prior-feature-mode`,
  `--result-boundary-target-amortized-prior-hidden-size`,
  `--result-boundary-target-amortized-prior-lr`,
  `--result-boundary-target-amortized-prior-min-entries`, and
  `--result-boundary-target-amortized-prior-replay-batch-size`.
- Added `scripts/diagnose_amortized_prior_from_trace.py` to fit the same prior
  from trace CSVs and evaluate train/heldout target generalization.
- Added a unit test that the prior can learn prompt-memory targets.

## Evidence

Trace source:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_src5000/2026-05-30_214753_463974_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-7006aed4f3/model-c-2digit-seed9
```

Diagnostic outputs:

```text
runs/amortized_prior_trace_diagnostics/heldout20_source_embedding_prior_fit.json
runs/amortized_prior_trace_diagnostics/heldout20_source_numeric_prior_fit.json
```

Results:

- The discovered train-memory labels were nearly correct:
  `memory_target_matches_true = 0.996875`.
- Embedding prior: train-memory fit `1.000`, train-vs-true `0.996875`,
  heldout-vs-true `0.0000`, confidence `0.7452`.
- Numeric prior: train-memory fit `1.000`, train-vs-true `0.996875`,
  heldout-vs-true `0.9125`, confidence `0.8486`.
- Tiny integrated smoke runs verified both online prior replay paths execute;
  those runs were not intended as performance gates.

## Interpretation

The arbitrary embedding prior simply memorizes prompt keys, so it repeats the
transductive-memory failure. Normalized numeric operand features are a real
fresh-prompt target-generalization signal: they recover most heldout targets
from answer-derived discovered train targets, without forced-scoring heldout
prompts.

This is not enough to claim the calculator source works on heldout prompts. The
next required gate is a full numeric-prior heldout source run that measures
whether detached prior pseudo-target replay lifts heldout calculator-result
accuracy above the no-prior `0.0875` boundary.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "amortized_prior"
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py ... --feature-mode embedding --steps 600
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py ... --feature-mode numeric --steps 2000
```
