# 2026-06-01 Target-Stratified Eval-Only Validation-Stop Gate

## Question

Did the validation-stop source miss because validation entries were excluded
from prior fitting, or because validation stopping itself is a bad signal?

## Implementation

- Added `--result-boundary-target-amortized-prior-fit-validation-mode`.
- Modes:
  - `holdout`: preserve existing behavior by excluding validation entries from
    prior-fit updates.
  - `eval_only`: fit on all prompt-memory entries and use the validation split
    only for validation metrics and stopping.
- Smoke-tested the new `eval_only` path:

```text
runs/smoke_target_stratified_prior_fit_eval_only_validation_stop/2026-06-01_163224_944795_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.003-uplr0.003-rbt1-zero_improvement-rbtt1-rbtchunk8-rbts4-rbtuniq-rbttopk2-rbton-169e29005b/model-c-2digit-seed13
```

The smoke completed, and final metrics recorded
`result_boundary_target_amortized_prior_fit_validation_mode = eval_only`.
The training curve included validation accuracy, update count, and stopped
fields.

## Fresh-Seed Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_val20_evalonly_stopval90pat100_src5000/2026-06-01_165015_552283_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-4d1509f598/model-c-2digit-seed13
```

Key changed arg versus validation-heldout:

```text
--result-boundary-target-amortized-prior-fit-validation-mode eval_only
```

Other key prior args:

```text
--result-boundary-target-amortized-prior-fit-sampling-mode target_stratified
--result-boundary-target-amortized-prior-fit-batch-size 160
--result-boundary-target-amortized-prior-fit-validation-fraction 0.2
--result-boundary-target-amortized-prior-stop-metric validation_accuracy
--result-boundary-target-amortized-prior-stop-train-accuracy 0.9
--result-boundary-target-amortized-prior-stop-patience 100
```

## Source Results

- Overall exact/calc `393/400 = 0.9825`.
- Train exact/calc `318/320 = 0.99375`.
- Heldout exact/calc `76/80 = 0.9500`.
- Train prior accuracy `0.978125`.
- Heldout prior accuracy `0.9500`.
- Prior updates `1613`.
- Fit stopped at step `3250` with validation accuracy `0.9821429` and
  `100` converged fit steps.
- Forced-result evals `124,416`.
- Heldout controls: injection-zero `0.0375`, forced-zero `0.0125`,
  forced-random `0.0125`.
- Prompt memory filled at step `100`.

## Fresh-Seed Trusted Additive Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_val20_evalonly_stopval90pat100_handoff600/2026-06-01_210232_382220_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed13
```

Results:

- Final eval `400/400 = 1.0000`.
- Final 128-sample controls:
  - injection-zero `0.0078125`
  - forced-zero `0.0390625`
  - forced-random `0.015625`
- Diagnostic exact/calc `1.0000` / `0.953125`.
- Routed hook calculator-result accuracies:
  - hook0 `0.9756`
  - hook1 `0.8649`
  - hook2 `1.0000`
  - hook3 `1.0000`

## Same-Seed Isolation Source Run

The fresh-seed run above used effective seed13 because the explicit
`--digits 2` CLI path offsets the base seed. To compare against the earlier
target-stratified seed11 benchmark, reran the source with base seed `9` and
`--digits 2`, producing `model-c-2digit-seed11`.

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_val20_evalonly_stopval90pat100_seed11_src5000/2026-06-01_211825_903000_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-4d1509f598/model-c-2digit-seed11
```

Results:

- Overall exact/calc `389/400 = 0.9725`.
- Train exact/calc `315/320 = 0.984375`.
- Heldout exact/calc `73/80 = 0.9125`.
- Train prior accuracy `0.9625`.
- Heldout prior accuracy `0.9125`.
- Prior updates `1784`.
- Fit stopped at step `3600` with validation accuracy `0.9833333` and
  `100` converged fit steps.
- Forced-result evals `89,088`.
- Heldout controls: injection-zero `0.0125`, forced-zero `0.0000`,
  forced-random `0.0000`.
- Prompt memory filled at step `100`.

## Same-Seed Trusted Additive Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_val20_evalonly_stopval90pat100_seed11_handoff600/2026-06-02_083125_007456_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Results:

- Final eval `400/400 = 1.0000`.
- Final 128-sample controls:
  - injection-zero `0.0000`
  - forced-zero `0.0703125`
  - forced-random `0.046875`
- Diagnostic exact/calc `1.0000` / `0.9453125`.
- Routed hook calculator-result accuracies:
  - hook0 `0.9143`
  - hook1 `0.9412`
  - hook2 `0.9333`
  - hook3 `1.0000`

## Interpretation

Eval-only validation stopping reverses the validation-heldout miss. The
validation signal is useful when it does not remove entries from the prior fit:
prior updates dropped below the sustained full-memory benchmark on both tested
effective seeds (`1613` and `1784` versus `1889`) and below target-stratified
every-2 (`2501`), while trusted non-bottleneck handoff cleared in both cases.

This is not the final scalable recipe. Forced-result evals rose to `124,416`,
above the previous target-stratified source's `67,584`; on the same seed11
isolation run they were `89,088`. The immediate cause is memory-fill timing:
target-stratified seed11 filled/froze prompt memory at step `50`, while both
eval-only runs filled at step `100`. The result reduces prior-fit optimizer
cost but does not yet reduce total source-acquisition cost.

## Steering

Make eval-only validation stopping the cost-reduction lead for the
target-stratified branch. Next work should stress it on a larger range and
diagnose memory-fill/forced-eval behavior before promoting it to the default
recipe. Do not run validation-heldout threshold/patience ladders.
