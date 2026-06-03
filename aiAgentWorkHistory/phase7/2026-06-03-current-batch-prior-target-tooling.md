# 2026-06-03 - Current-Batch Prior Target Tooling

## Purpose

Add a shared-prior target-supply path that does not write pseudo-targets into
prompt memory and does not rely on a separately sampled replay pool. The
previous route-excluded variants either replayed the prior on sampled pools,
bootstrapped prompt-memory entries, or trained the prior from candidate
evidence; none directly supplied prior targets to the live current batch.

## Implementation

- Added `result_boundary_amortized_prior_current_batch_loss`.
- Added `--result-boundary-target-amortized-prior-current-batch-weight`.
- Added `--result-boundary-target-amortized-prior-current-batch-routes`.
- Added `--result-boundary-target-amortized-prior-current-batch-min-confidence`.
- The helper asks the shared amortized prior for detached result targets on the
  current training batch, optionally filters by routed hook id and confidence,
  and adds result-logit CE only on selected examples.
- It does not add prompt-memory entries, score new candidates, or sample a
  separate replay batch.
- Added per-step metrics for objective, loss, selected count/fraction,
  route-filter fraction, pseudo-accuracy, and confidence.
- Added a focused unit test verifying route filtering and confidence gating.

## Smoke

Ran a five-step 2-digit answer-decoder smoke with route `1` current-batch prior
targets enabled:

```text
runs/codex_smoke_current_batch_prior/2026-06-02_201642_616704_model-c-op0-19-fullgrid-streamb8-heldout0.25-gumbel_concrete_interface-result_space-inlr0.03-uplr0.0003-rbt0.1-hard_best_result-rbtt1-rbtchunk64-rbts2-rbtuniq-rbtonlineh-aff7f9ea18/model-c-2digit-seed2
```

Smoke result:

- Final exact-match was `0/16`; not a source-quality test.
- The current-batch objective fired on both logged steps.
- Step `0`: objective `0.3661`, selected count `2`, route fraction `0.25`,
  pseudo-accuracy `0.0`.
- Step `5`: objective `0.3689`, selected count `3`, route fraction `0.375`,
  pseudo-accuracy `0.3333`.
- Final metrics recorded the new config fields.

## Verification

```bash
python3 -m py_compile scripts/overfit_one_batch.py
python3 -m pytest tests/test_model.py -k "amortized_prior or candidate_evidence or current_batch_prior or route_exclusion"
git diff --check
```

The focused pytest slice passed with `5 passed, 156 deselected`.

## Next

Use this as the first concrete test of direct shared-prior target supply on a
real source gate. Require heldout prompt quality and excluded-route quality
before any trusted handoff. Do not treat the smoke as evidence of source
quality.
