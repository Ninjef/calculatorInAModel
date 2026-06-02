# 2026-06-02 - Route-excluded target discovery tooling

## Question

Can we turn the route-heldout shared-prior diagnostic into an actual source
training gate by disabling direct prompt-memory target discovery on specified
routed calculator hooks?

## Implementation

- Added `--result-boundary-target-memory-update-exclude-routes`, a
  comma-separated list of route ids whose prompt-memory target discovery is
  skipped.
- Added route-aware prompt-memory eligibility helpers so the prompt-memory
  expected-full count is computed from score-eligible routes only.
- Threaded the route exclusion into `result_boundary_prompt_hard_memory_loss`.
  Excluded routes can still be trained by amortized-prior replay because replay
  samples from train/heldout prompt pools globally.
- Logged
  `result_boundary_target_prompt_memory_score_eligible_fraction` and
  `result_boundary_target_prompt_memory_update_excluded_fraction` in training
  curves.
- Recorded the exclusion string in `config.json` and run suffixes.

## Verification

Focused regression:

```bash
python3 -m pytest tests/test_model.py -k "prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior"
```

Result: `4 passed, 151 deselected`.

Syntax:

```bash
python3 -m py_compile scripts/overfit_one_batch.py
```

Result: passed.

CLI smoke:

```text
runs/2026-06-02_route_exclusion_smoke/2026-06-02_154720_202336_model-c-op0-2-fullgrid-streamb4-gumbel_concrete_interface-result_space-rbt1-zero_improvement-rbtt1-rbtchunk2-rbts1-rbtuniq-rbttopk1-rbtonlinehardmem-rbtmemprompt-rbtmemf-97315c88e5/model-c-2digit-seed2
```

The smoke wrote `config.json` with
`result_boundary_target_memory_update_exclude_routes="1"` and computed prompt
memory expected entries as `6`, matching op0-2 with one of the three active
`left_operand_mod` routes withheld from direct target discovery.

Full regression was also started:

```bash
python3 -m pytest tests/test_model.py -q
```

## Interpretation

This is enabling tooling, not a solved training result. It creates the missing
mechanism for the next high-leverage gate: train a routed source where one or
more routes get calculator-result pressure only through shared/global numeric
prior replay, then evaluate source heldout quality and trusted frozen-policy
additive handoff.

## Next

Run the actual shared-prior source gate. Do not treat more cap/seed/fit ladders
as algorithmic progress unless they change how target discovery is shared or
removed.
