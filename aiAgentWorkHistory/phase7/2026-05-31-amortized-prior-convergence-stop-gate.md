# 2026-05-31 - Amortized Prior Convergence Stop Gate

## Question

Can we stop full-memory numeric-prior fitting after the prompt memory and prior
have converged, preserving the every-2 heldout source/handoff result with fewer
than `2501` prior updates?

## Code Change

Added two options to `scripts/overfit_one_batch.py`:

- `--result-boundary-target-amortized-prior-stop-train-accuracy`
- `--result-boundary-target-amortized-prior-stop-patience`

The stop rule is active only after prompt memory is full. After fitting stops,
train and heldout replay continue to use the latest prior, but the expensive
full-memory prior optimizer/evaluation path is skipped.

New metrics:

- `result_boundary_target_amortized_prior_fit_converged_steps`
- `result_boundary_target_amortized_prior_fit_stopped`
- `result_boundary_target_amortized_prior_memory_full`

Smoke:

```text
runs/smoke_amortized_prior_stop_fit/2026-05-31_132837_802776_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-rbt1-zero_improvement-rbtt1-rbtchunk8-rbts4-rbtuniq-rbttopk2-rbtonlinehardmem-rbtmempr-e56a33a2e0/model-c-2digit-seed9
```

## First-Hit Convergence Stop

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1_src5000/2026-05-31_132937_811869_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rb-ce92d753c1/model-c-2digit-seed9
```

Key args:

```text
--result-boundary-target-amortized-prior-fit-every 2
--result-boundary-target-amortized-prior-stop-train-accuracy 1.0
--result-boundary-target-amortized-prior-stop-patience 1
```

Results:

- Overall exact/calc `393/400 = 0.9825`.
- Train exact/calc `320/320 = 1.0000`.
- Heldout exact/calc `70/80 = 0.8750`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prior updates `1029`.
- Prior train/heldout accuracy `1.0000` / `0.8750`.

Interpretation: first-hit train-memory convergence stops too early. It saves
updates, but the numeric prior has not stabilized enough for heldout prompts.

## Sustained Convergence Stop

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_src5000/2026-05-31_134344_475019_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-a869378bc1/model-c-2digit-seed9
```

Key args:

```text
--result-boundary-target-amortized-prior-fit-every 2
--result-boundary-target-amortized-prior-stop-train-accuracy 1.0
--result-boundary-target-amortized-prior-stop-patience 100
```

Results:

- Overall exact/calc `398/400 = 0.9950`.
- Train exact/calc `320/320 = 1.0000`.
- Heldout exact/calc `73/80 = 0.9125`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prior updates `1889`, down from every-2 `2501` and full-fit `5001`.
- Prior train/heldout accuracy `1.0000` / `0.9125`.
- Forced-result evals stayed `86,016`.

## Trusted Handoff

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_handoff600/2026-05-31_135657_342042_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Results:

- Final eval `397/400 = 0.9925`.
- Diagnostic exact/calc `127/128 = 0.9921875` / `0.984375`.
- Final 128-sample controls: injection-zero `0.0546875`, forced-zero
  `0.0078125`, forced-random `0.0078125`.
- Routed hook calculator-result accuracy: hook0 `0.9574`, hook1 `1.0000`,
  hook2 `1.0000`, hook3 `1.0000`.

## Conclusion

Sustained train-memory convergence gating is a real cost reducer: it preserves
the heldout source and trusted non-bottleneck handoff while cutting prior
updates to `1889`.

First-hit convergence is negative and should not be repeated. Patience-100 is
the safe train-convergence benchmark, but the next high-leverage step should
not be a patience ladder. Use a validation/heldout-prior stopping signal or
coreset/reservoir prior fitting to push below `1889` updates while preserving
the same source/handoff gate.
