# 2026-06-02 - Op29 Eval-Only Target-Stratified Prior Stress

## Question

Does the op19 eval-only target-stratified numeric-prior recipe scale to
`operand_max=29` with the same constant prior fit batch, or does the prior
itself become the scaling bottleneck?

## Pre-Run Check

Read `CLAUDE.md`, `RESEARCH_STATE.md`, and `HYPOTHESIS_LEDGER.md`. Searched
research memory for op29 amortized-prior / target-stratified / heldout scaling
and found no prior op29 numeric-prior heldout result. The closest memories were
the earlier op29 forced-margin range stress and the op19 target-stratified /
numeric-prior results.

## Source Run

Run directory:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_fit160_targetstrat_val20_evalonly_stopval90pat100_src5000/2026-06-02_095629_230652_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-c46e34c022/model-c-2digit-seed9
```

Key configuration:

```text
--operand-max 29
--streaming-train-batch-size 64
--streaming-train-heldout-fraction 0.2
--calculator-operand-vocab-size 30
--calculator-hook-count 4
--calculator-hook-routing left_operand_mod
--share-calculator-output-proj
--result-boundary-target-online-hard-memory
--result-boundary-target-online-memory-key-mode prompt
--result-boundary-target-online-memory-freeze-when-full
--result-boundary-target-amortized-prior-feature-mode numeric
--result-boundary-target-amortized-prior-hidden-size 64
--result-boundary-target-amortized-prior-fit-batch-size 160
--result-boundary-target-amortized-prior-fit-sampling-mode target_stratified
--result-boundary-target-amortized-prior-fit-validation-fraction 0.2
--result-boundary-target-amortized-prior-fit-validation-mode eval_only
--result-boundary-target-amortized-prior-fit-every 2
--result-boundary-target-amortized-prior-stop-metric validation_accuracy
--result-boundary-target-amortized-prior-stop-train-accuracy 0.9
--result-boundary-target-amortized-prior-stop-patience 100
```

## Source Results

- Overall exact/calc: `866/900 = 0.9622`.
- Train exact/calc: `0.9931`.
- Heldout exact/calc: `0.8444`.
- Train prior accuracy: `0.8375`.
- Heldout prior accuracy: `0.7667`.
- Prior updates: `2501`; validation stop did not fire.
- Prompt memory filled at step `200` with `720/720` entries.
- Forced-result evals: `290,304`.
- Heldout controls: injection-zero `0.0278`, forced-zero `0.0000`,
  forced-random `0.0111`.

No trusted handoff was run because the source missed the heldout gate.

## Post-Hoc Prior Diagnostics

The source emitted `train_prompt_trace_rows.csv` and
`heldout_prompt_trace_rows.csv`, so I used
`scripts/diagnose_amortized_prior_from_trace.py` to separate target quality
from prior fit quality.

The train-memory targets were mostly correct:
`memory_target_matches_true = 0.9931`.

Results:

| Diagnostic | Train true acc | Heldout true acc | Memory fit acc |
| --- | ---: | ---: | ---: |
| h64 numeric, full memory, `600` steps | `0.8958` | `0.6722` | `0.9000` |
| h64 numeric, full memory, `2500` steps | `0.9875` | `0.9000` | `0.9889` |
| h128 numeric, full memory, `2500` steps | `0.9889` | `0.9278` | `0.9931` |

Diagnostic outputs were saved next to the run as:

```text
posthoc_full_memory_numeric_h64_prior_diag.json
posthoc_full_memory_numeric_h64_steps2500_prior_diag.json
posthoc_full_memory_numeric_h128_steps2500_prior_diag.json
```

## Interpretation

This is a mixed negative for the constant-batch op29 scaling story.

The calculator path is causal and the online memory targets are good, but the
constant target-stratified fit batch `160` is not enough for op29 prior
generalization. Longer full-memory h64 fitting and h128 capacity both recover
substantial heldout accuracy post-hoc, so the next bottleneck is prior
capacity/optimization and scalable fit dynamics, not memory fill, target
quality, or calculator wiring.

## Anti-Rerun Guidance

Do not repeat this as another op29 `fit_batch_size=160` run, validation
threshold/patience ladder, or random batch-size ladder. Also do not run a
trusted handoff from this source as progress.

The next allowed test should materially change the prior mechanism or fit
dynamics, such as richer numeric features/capacity, post-memory-fill full
refresh, or a coverage-aware/proportional fit with explicit cost accounting.
