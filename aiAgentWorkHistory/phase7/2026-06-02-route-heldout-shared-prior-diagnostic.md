# 2026-06-02 Route-Heldout Shared Prior Diagnostic

## Question

Can a single structured numeric prior learn target behavior from some routed
calculators and generalize to an unscored routed calculator?

This is a cheap diagnostic for breaking per-calculator target/prior scaling. It
is not yet a source-training run that skips target discovery on some routes.

## Code

Extended `scripts/diagnose_amortized_prior_from_trace.py`:

- Added `--split-mode route_heldout`.
- Added `--heldout-routes`.
- In route-heldout mode, fit memory is built from `train_prompt_trace_rows.csv`
  excluding the heldout `calculator_hook_route`; evaluation is on the withheld
  route's train-trace rows.

## Trace

Source trace:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_qcap2000_dualstop_val90_train98_pat100_src5000/2026-06-02_143237_450578_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-b3b0395c67/model-c-2digit-seed11
```

## Commands

Numeric h128 route-heldout diagnostics:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py <source-trace> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode numeric --hidden-size 128 --steps 2500 --split-mode route_heldout --heldout-routes 0 --output runs/route_heldout_prior_diag/op29_cap_seed11_route0.json
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py <source-trace> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode numeric --hidden-size 128 --steps 2500 --split-mode route_heldout --heldout-routes 1 --output runs/route_heldout_prior_diag/op29_cap_seed11_route1.json
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py <source-trace> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode numeric --hidden-size 128 --steps 2500 --split-mode route_heldout --heldout-routes 2 --output runs/route_heldout_prior_diag/op29_cap_seed11_route2.json
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py <source-trace> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode numeric --hidden-size 128 --steps 2500 --split-mode route_heldout --heldout-routes 3 --output runs/route_heldout_prior_diag/op29_cap_seed11_route3.json
```

Embedding control on route 0:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_amortized_prior_from_trace.py <source-trace> --operand-vocab-size 30 --result-vocab-size 59 --feature-mode embedding --hidden-size 128 --steps 2500 --split-mode route_heldout --heldout-routes 0 --output runs/route_heldout_prior_diag/op29_cap_seed11_route0_embedding.json
```

## Results

| Heldout route | Fit rows | Heldout rows | Prior type | Train accuracy | Heldout-route accuracy |
| ---: | ---: | ---: | --- | ---: | ---: |
| `0` | `510` | `210` | numeric h128 | `0.9784` | `0.9333` |
| `1` | `499` | `221` | numeric h128 | `0.9960` | `0.9683` |
| `2` | `575` | `145` | numeric h128 | `0.9983` | `0.9793` |
| `3` | `576` | `144` | numeric h128 | `0.9931` | `0.9583` |
| `0` | `510` | `210` | embedding h128 | `1.0000` | `0.0000` |

All fit-memory targets matched the true sum (`1.0000`) in the numeric and
embedding runs.

## Interpretation

Result:

```text
route_heldout_numeric_prior_shared_target_positive
```

The numeric prior can infer the target function for an unscored route from the
other three routed calculators. The embedding control memorizes train routes and
fails completely on the heldout route, so the numeric result is structured
sharing rather than generic prompt memory.

This is the first concrete positive for breaking per-calculator target/prior
scaling in the current family. It remains a diagnostic: the source model has
not yet been trained with target discovery disabled or reduced for some routes,
and no trusted handoff exists for that setting.

Next test: train a routed source where sparse target discovery is intentionally
disabled or heavily reduced for one or more routes, while a shared/global
numeric prior supplies replay targets to all routes. Then run the trusted
frozen-policy additive handoff gate.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_amortized_prior_trace_diagnostic.py tests/test_prior_replay_scaling.py -q
python3 -m py_compile scripts/diagnose_amortized_prior_from_trace.py
```

Results:

```text
8 passed in 0.02s
py_compile passed
```
