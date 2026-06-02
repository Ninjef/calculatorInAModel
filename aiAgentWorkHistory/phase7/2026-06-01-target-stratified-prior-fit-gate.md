# 2026-06-01 - Target-Stratified Prior Fit Gate

## Question

Can a structured half-memory numeric-prior fit batch preserve the integrated
amortized-prior source and non-bottleneck handoff gates after random
half-memory prior fits failed?

## Code Change

Added:

```text
--result-boundary-target-amortized-prior-fit-sampling-mode
```

Modes:

- `random`: existing with-replacement minibatch behavior.
- `target_stratified`: when prior fit batch size is smaller than prompt memory,
  sample a balanced batch across discovered target result classes.

The new mode is recorded in config, metrics, run suffixes, and startup logs.

Smoke:

```text
runs/smoke_target_stratified_prior_fit/2026-06-01_094030_345614_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.003-uplr0.003-rbt1-zero_improvement-rbtt1-rbtchunk8-rbts4-rbtuniq-rbttopk2-rbton-d286602445/model-c-2digit-seed11
```

## Source Gate

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_every2_stop1pat100_src5000/2026-06-01_094407_472913_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-2e08204a5e/model-c-2digit-seed11
```

Changed only the prior fit sampling mode versus the random half-memory negative:

```text
--result-boundary-target-amortized-prior-fit-batch-size 160
--result-boundary-target-amortized-prior-fit-sampling-mode target_stratified
--result-boundary-target-amortized-prior-fit-every 2
--result-boundary-target-amortized-prior-stop-train-accuracy 1.0
--result-boundary-target-amortized-prior-stop-patience 100
```

Results:

- Overall exact/calc: `396/400 = 0.9900`.
- Train exact/calc: `319/320 = 0.996875`.
- Heldout exact/calc: `75/80 = 0.9375`.
- Heldout controls: injection-zero `0.0125`, forced-zero `0.0000`,
  forced-random `0.0000`.
- Prior train/heldout accuracy: `0.965625` / `0.9000`.
- Prior updates: `2501`; stop rule did not activate.
- Forced-result evals: `67,584`, lower than the prior random-half run's
  `86,016`.

Comparison:

| Prior fit | Source overall | Heldout exact/calc | Prior train/heldout | Updates | Forced evals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random half-memory | `0.9675` | `0.8125` | `0.9094 / 0.7750` | `2501` | `86,016` |
| Full-memory every-2 patience-100 | `0.9950` | `0.9125` | `1.0000 / 0.9125` | `1889` | `86,016` |
| Target-stratified half-memory | `0.9900` | `0.9375` | `0.9656 / 0.9000` | `2501` | `67,584` |

## Trusted Additive Handoff

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fit160_targetstrat_every2_stop1pat100_handoff600/2026-06-01_131137_211268_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Settings:

- Loaded the target-stratified source final checkpoint with compatible loading.
- Switched to non-bottleneck `calculator_bottleneck_mode=none`.
- Used `calculator_estimator=ste`.
- Froze the calculator policy.
- Trained downstream/readout for `600` steps.

Results:

- Final eval: `399/400 = 0.9975`.
- Final 128-sample counterfactual controls from `metrics["counterfactuals"]`:
  injection-zero `0.0000`, forced-zero `0.0546875`, forced-random `0.0390625`.
- Diagnostic exact/calc: `127/128 = 0.9921875` / `1.0000`.
- Routed hook calculator-result accuracy: all four hooks `1.0000`.

## Interpretation

Target-stratified half-memory prior fitting is the first positive structured
coreset result in this branch. It reverses the random-half failure, preserves
the heldout source gate, and transfers causally into the trusted
non-bottleneck frozen-policy handoff.

This improves the scalability story, but it is not the full solution:

- Prior updates did not drop below `2501`; the convergence stop never fired.
- The method still depends on sparse answer-derived forced-result scoring to
  fill prompt memory.
- The source/handoff result should be replicated on a fresh seed or combined
  with a convergence/validation stop before treating it as the new default.

## Next

Use `target_stratified` as the structured-coreset benchmark. Do not run random
fit-batch-size ladders. The next high-leverage step is to combine
target-stratified fit batches with a validation/prior convergence stop or to
stress the method on a fresh seed / range axis.
