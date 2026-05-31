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
- Added `scripts/run_amortized_prior_replay_gate.py` to load the heldout-failed
  source, fit the prior, replay pseudo-targets into the source result head, and
  measure train/heldout uptake.
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
runs/amortized_prior_replay_gates/heldout20_numeric_result_head_500/summary.json
runs/amortized_prior_replay_gates/heldout20_numeric_result_head_trainmix500/summary.json
runs/amortized_prior_replay_gates/heldout20_embedding_result_head_trainmix500/summary.json
```

Results:

- The discovered train-memory labels were nearly correct:
  `memory_target_matches_true = 0.996875`.
- Embedding prior: train-memory fit `1.000`, train-vs-true `0.996875`,
  heldout-vs-true `0.0000`, confidence `0.7452`.
- Numeric prior: train-memory fit `1.000`, train-vs-true `0.996875`,
  heldout-vs-true `0.9125`, confidence `0.8486`.
- Heldout-only numeric result-head replay moved heldout exact/calc from
  `0.0875` to `0.9000`, but damaged train exact/calc to `0.365625`.
- Mixed train+heldout numeric result-head replay moved heldout exact/calc from
  `0.0875` to `0.9125` while preserving train at `0.990625`.
- Mixed train+heldout embedding-prior replay reached only `0.0125` heldout and
  `0.959375` train, confirming that numeric features supply the useful
  fresh-prompt target signal.

## Interpretation

The arbitrary embedding prior simply memorizes prompt keys, so it repeats the
transductive-memory failure. Normalized numeric operand features are a real
fresh-prompt target-generalization signal: they recover most heldout targets
from answer-derived discovered train targets, without forced-scoring heldout
prompts. Post-hoc replay shows the heldout-failed source result head can absorb
those pseudo-targets and repair heldout calculator use without sacrificing seen
prompts.

This is still not enough to claim end-to-end source training works on heldout
prompts. The next required gate is to integrate numeric-prior replay during the
streaming source run itself, then measure seen and heldout calculator accuracy
before any trusted handoff.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "amortized_prior"
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py ... --feature-mode embedding --steps 600
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py ... --feature-mode numeric --steps 2000
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_amortized_prior_replay_gate.py ... --prior-feature-mode numeric --train-replay-weight 1
```
