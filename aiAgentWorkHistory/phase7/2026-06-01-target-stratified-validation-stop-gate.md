# 2026-06-01 Target-Stratified Validation-Stop Gate

## Question

Can a validation-aware stop reduce target-stratified amortized-prior fitting
below `2501` updates without losing the heldout prompt source gate?

## Implementation

- Added `--result-boundary-target-amortized-prior-fit-validation-fraction`.
- Added `--result-boundary-target-amortized-prior-stop-metric` with
  `train_accuracy` and `validation_accuracy`.
- Added deterministic prompt-memory validation masking.
- When validation fraction is nonzero, the prior fits only non-validation
  entries and reports validation accuracy on the held-out memory entries.

Smoke:

```text
runs/smoke_target_stratified_prior_fit_validation_stop/2026-06-01_132615_812985_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.003-uplr0.003-rbt1-zero_improvement-rbtt1-rbtchunk8-rbts4-rbtuniq-rbttopk2-rbton-e7b4dd34ff/model-c-2digit-seed11
```

The smoke completed and logged validation fraction, stop metric, and
validation accuracy fields.

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_val20_stopval90pat100_src5000/2026-06-01_132747_335760_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-91ccc48399/model-c-2digit-seed11
```

Key changed args:

```text
--result-boundary-target-amortized-prior-fit-sampling-mode target_stratified
--result-boundary-target-amortized-prior-fit-batch-size 160
--result-boundary-target-amortized-prior-fit-validation-fraction 0.2
--result-boundary-target-amortized-prior-stop-metric validation_accuracy
--result-boundary-target-amortized-prior-stop-train-accuracy 0.9
--result-boundary-target-amortized-prior-stop-patience 100
```

## Results

- Overall exact/calc `389/400 = 0.9725`.
- Train exact/calc `317/320 = 0.990625`.
- Heldout exact/calc `69/80 = 0.8625`.
- Train prior accuracy `0.98125`.
- Heldout prior accuracy `0.8625`.
- Prior updates `2359`.
- Forced-result evals `53,760`.
- Heldout controls: injection-zero `0.0125`, forced-zero `0.0000`,
  forced-random `0.0000`.
- At step `4750`, validation accuracy was `0.9333333`,
  converged steps were `100`, and fitting stopped. It stayed stopped through
  step `5000`.

## Interpretation

The validation stop reduced cost only modestly and missed the heldout source
gate. Compared with the target-stratified source benchmark, prior updates fell
from `2501` to `2359`, but heldout exact/calc fell from `0.9375` to `0.8625`.
It also underperformed the full-memory sustained-convergence benchmark
heldout `0.9125`.

The likely failure is that holding out `20%` of prompt-memory entries from
prior fitting weakens the numeric prior and replay target quality. Validation
accuracy over held-out memory entries did not predict true heldout prompt
quality well enough for stopping.

No trusted additive handoff was run because the source missed the heldout
gate.

## Steering

Do not run validation-heldout threshold/patience ladders as novelty. If using
a validation signal next, make it eval-only while fitting all entries, or use a
rolling/full-fit target-stratified stopping signal. Otherwise stress the
positive target-stratified coreset on a fresh seed or range axis.
