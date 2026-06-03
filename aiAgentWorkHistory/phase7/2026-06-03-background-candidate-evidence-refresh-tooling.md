# 2026-06-03 - Background Candidate-Evidence Refresh Tooling

## Purpose

Add a shared/global target-formation path after the route-excluded tweak branch
closed. The prior route-excluded failures all depended on a prompt-memory target
table filling early, after which the amortized prior mostly replayed or copied
already frozen targets. The new path periodically scores fresh train-pool
prompts into the shared prior without writing prompt-memory entries.

## Implementation

- Added `--result-boundary-target-amortized-prior-evidence-refresh-weight`.
- Added `--result-boundary-target-amortized-prior-evidence-refresh-batch-size`.
- Added `--result-boundary-target-amortized-prior-evidence-refresh-every`.
- Added `--result-boundary-target-amortized-prior-evidence-refresh-exclude-routes`.
- Added `train_result_boundary_amortized_prior_from_scored_candidates`.
- The helper samples/scans candidate result classes using the same
  zero-improvement semantics as prompt hard memory, then trains only the shared
  amortized prior.
- It does not add or update prompt-memory entries.
- Route exclusion applies only to evidence scoring; prior replay may still
  train excluded routes from the shared prior.
- Added separated final counters for evidence-refresh updates, examples, and
  forced-result evals.
- Added a focused unit test for route-excluded refresh accounting.

## Smoke

Ran a tiny route-excluded smoke with route `1` excluded from refresh scoring:

```text
runs/2026-06-03_prior_evidence_refresh_smoke_final/2026-06-02_193543_504098_model-c-op0-2-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk16-rbts4-rbtuniq-rbttopk2-rbton-b987cd16ea/model-c-2digit-seed15
```

Smoke result:

- Final exact-match was `0/20`; not a source-quality test.
- Evidence-refresh updates/examples: `2/5`.
- Evidence-refresh forced evals: `76`.
- Candidate-evidence generic counters matched the refresh counters because
  direct prompt-memory candidate evidence was disabled.
- Final metrics and training-curve rows included refresh fields.

## Verification

```bash
python3 -m py_compile scripts/overfit_one_batch.py
python3 -m pytest tests/test_model.py -k "candidate_evidence_refresh or candidate_evidence_prior_update or prior_bootstrap_memory or subset_arithmetic_batch_by_routes or prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior or route_exclusion"
```

The focused pytest slice passed with `9 passed, 151 deselected`.

## Next

Run a real op19 route-excluded source with background evidence refresh enabled,
excluding route `1` from evidence scoring while allowing prior replay to train
all routes. Judge only on heldout prompt and excluded-route quality before any
trusted handoff.
