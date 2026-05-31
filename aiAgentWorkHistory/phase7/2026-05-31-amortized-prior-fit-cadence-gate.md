# 2026-05-31 - Amortized Prior Fit Cadence Gate

## Question

Can lower-frequency full-memory fitting of the numeric amortized prior preserve
the integrated heldout source and trusted handoff result while reducing prior
optimizer updates?

## Code Change

Added `--result-boundary-target-amortized-prior-fit-every` to
`scripts/overfit_one_batch.py`.

- Default `1` preserves prior behavior.
- The first eligible prior fit always runs.
- Later fits happen every `N` steps, while train/heldout replay still uses the
  latest prior.
- Added `result_boundary_target_amortized_prior_fit_step` to distinguish prior
  update steps from replay-only steps.

Smoke:

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py ... --steps 2 --result-boundary-target-amortized-prior-fit-batch-size 0 --result-boundary-target-amortized-prior-fit-every 10 --result-boundary-target-amortized-prior-train-replay-weight 1 --run-root runs/smoke_amortized_prior_fit_every
```

Run:

```text
runs/smoke_amortized_prior_fit_every/2026-05-31_125322_862659_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-rbt1-zero_improvement-rbtt1-rbtchunk8-rbts4-rbtuniq-rbttopk2-rbtonlinehardmem-rbtmempr-40ddd0d949/model-c-2digit-seed9
```

The smoke exercised full-memory prior fit, train replay, and heldout replay and
wrote the new fit-step metric.

## Every-10 Source

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every10_src5000/2026-05-31_125426_550266_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-7a4bd26fd0/model-c-2digit-seed9
```

Command difference from the full-fit benchmark:

```text
--result-boundary-target-amortized-prior-fit-every 10
```

Results:

- Overall exact/calc `379/400 = 0.9475`.
- Train exact/calc `313/320 = 0.978125`.
- Heldout exact/calc `61/80 = 0.7625`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prior updates `501`.
- Prior train/heldout accuracy `0.953125` / `0.7875`.
- Prompt-memory entries `320`; forced-result evals `107,520`.

Interpretation: every-10 cadence underfits the prior. The heldout miss is not a
handoff issue, so no trusted handoff was run from this source.

## Every-2 Source

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_src5000/2026-05-31_130618_663672_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-1c87b969de/model-c-2digit-seed9
```

Command difference from the full-fit benchmark:

```text
--result-boundary-target-amortized-prior-fit-every 2
```

Results:

- Overall exact/calc `398/400 = 0.9950`.
- Train exact/calc `320/320 = 1.0000`.
- Heldout exact/calc `73/80 = 0.9125`.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Prior updates `2501` versus `5001` in the full-fit benchmark.
- Prior train/heldout accuracy `1.0000` / `0.9125`.
- Prompt-memory entries `320`; forced-result evals `86,016`.

## Trusted Handoff From Every-2 Source

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_handoff600/2026-05-31_131746_414262_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Results:

- Final eval `395/400 = 0.9875`.
- Diagnostic exact/calc `125/128 = 0.9765625` / `0.984375`.
- Final 128-sample controls: injection-zero `0.015625`, forced-zero
  `0.0078125`, forced-random `0.0078125`.
- Routed hook calculator-result accuracy: hook0 `0.9574`, hook1 `1.0000`,
  hook2 `1.0000`, hook3 `1.0000`.

## Conclusion

Every-other-step full-memory prior fitting preserves the heldout source gate
and trusted non-bottleneck handoff with a 2x reduction in prior updates.
Every-10 fitting fails by prior underfitting.

Do not run a cadence ladder as novelty. The next high-leverage experiment is a
convergence-gated or coreset/reservoir prior fit that beats `2501` updates
while preserving every-2 source and handoff quality.
