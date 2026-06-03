# 2026-06-03 - Candidate-Evidence Prior Tooling

## Purpose

Wire the next route-excluded shared-prior mechanism after route replay and
prior-bootstrap prompt memory both failed to clear the op19 source gate. The
new path trains the shared amortized prior directly from candidate-scored target
evidence at discovery time, before prompt memory freezes and before an excluded
route has to rely on late pseudo-labels.

## Implementation

- Added `--result-boundary-target-amortized-prior-candidate-evidence-weight`.
- Added `train_result_boundary_amortized_prior_on_candidate_evidence`.
- The helper trains the existing `ResultBoundaryAmortizedPrior` model on
  already-scored positive candidate targets, weighted by the new flag.
- `result_boundary_prompt_hard_memory_loss` now calls the helper when the flag
  is positive and the amortized prior is enabled.
- Added runtime/final counters for candidate-evidence updates and examples,
  plus per-update loss, objective, count, target-vs-true accuracy, confidence,
  and prior-vs-target accuracy.
- Added parser validation so the weight must be non-negative and requires an
  active amortized-prior objective.
- Added a focused unit test for accounting and target accuracy.

## Smoke

Ran a tiny op2 route-excluded smoke to confirm the live training loop records
candidate-evidence updates:

```text
runs/2026-06-03_prior_candidate_evidence_smoke/2026-06-02_185909_784821_model-c-op0-2-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk16-rbts4-rbtuniq-rbttopk2-rbton-f447dfc864/model-c-2digit-seed10
```

Smoke result:

- Final exact-match was `13/40 = 0.325`; this was not a source-gate run.
- Final candidate-evidence counters were `27` prior updates over `81` examples.
- Training-curve rows recorded nonzero candidate-evidence batches; step `20`
  had `6` evidence targets, target-vs-true accuracy `1.0`, `19` cumulative
  updates, and `60` cumulative examples.
- Final forced-result evals were `668`, from the configured candidate scorer.
  The candidate-evidence update itself reuses those scored targets.

## Verification

```bash
python3 -m py_compile scripts/overfit_one_batch.py
python3 -m pytest tests/test_model.py -k "candidate_evidence_prior_update or prior_bootstrap_memory or subset_arithmetic_batch_by_routes or prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior or route_exclusion"
```

The focused pytest slice passed with `8 passed, 151 deselected`.

## Next

Run a real op19 route-excluded source gate with
`--result-boundary-target-amortized-prior-candidate-evidence-weight` enabled.
Judge it on heldout prompt quality and excluded-route quality before any trusted
handoff. Do not count the tiny smoke as evidence of source-quality improvement.
