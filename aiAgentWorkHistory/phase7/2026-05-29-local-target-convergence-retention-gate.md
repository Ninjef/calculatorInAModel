# 2026-05-29 - Local-Target Convergence And Retention Gate

## Question

Does the promising `policy_reweighted_t1` local-target branch from the 200-step
Stage 1 lift gate keep improving with a longer budget, and does ordinary
answer-only training retain the learned natural result-level calculator policy?

This extends the prior gate rather than repeating it:

- Prior gate: 200 target-training steps, four branches.
- This gate: 800 target-training steps plus 200 answer-only retention steps,
  focused on `hard_boundary` vs `policy_reweighted_t1`.

## Code

Extended:

```text
scripts/run_phase7_local_target_stage1_lift_gate.py
```

The runner now supports an optional answer-only retention phase via:

```text
--retention-steps
--retention-eval-every
--retention-lr
--retention-input-proj-lr
--retention-upstream-lr
```

Rows include `phase` and `phase_step` so target-training and retention curves
are separable.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_local_target_convergence_retention_gate/smoke_op3
```

Focused gate:

```text
runs/2026-05-29_phase7_local_target_convergence_retention_gate/policy_t1_vs_hard_steps800_retention200
```

Command:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 800 \
  --retention-steps 200 \
  --eval-every 100 \
  --retention-eval-every 50 \
  --control-eval-every 400 \
  --eval-samples 128 \
  --branches hard_boundary,policy_reweighted_t1 \
  --output-root runs/2026-05-29_phase7_local_target_convergence_retention_gate/policy_t1_vs_hard_steps800_retention200
```

## Results

Final printed summary:

```text
hard_boundary: target_normal=0.7656 target_calc=0.8200 retention_normal=0.8281 retention_calc=0.8050 best_normal=0.8281
policy_reweighted_t1: target_normal=0.6875 target_calc=0.7050 retention_normal=0.8750 retention_calc=0.8925 best_normal=0.8750
```

Exact-grid calculator-result curve:

| Branch | Step 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 target | 1000 retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard_boundary` | 0.0100 | 0.1925 | 0.5500 | 0.7600 | 0.8600 | 0.9225 | 0.7325 | 0.7175 | 0.8200 | 0.8050 |
| `policy_reweighted_t1` | 0.0100 | 0.1700 | 0.5600 | 0.7100 | 0.7475 | 0.8275 | 0.8925 | 0.8375 | 0.7050 | 0.8925 |

Retention checkpoints:

| Branch | Retention step 50 | Retention step 100 | Retention step 150 | Retention step 200 |
| --- | ---: | ---: | ---: | ---: |
| `hard_boundary` calc | 0.7675 | 0.8200 | 0.7150 | 0.8050 |
| `policy_reweighted_t1` calc | 0.8325 | 0.8450 | 0.6950 | 0.8925 |

Final sampled controls at retention step 200:

| Branch | Normal | Calculator result | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard_boundary` | 0.8281 | 0.8281 | 0.0234 | 0.0156 | 1.0000 |
| `policy_reweighted_t1` | 0.8750 | 0.8750 | 0.0234 | 0.0156 | 1.0000 |

## Interpretation

This is a retention-positive but nonmonotonic result.

`policy_reweighted_t1` does not dominate throughout target training. It trails
hard-boundary at target step 800 (`0.7050` vs `0.8200` exact-grid calc), even
though it peaks earlier at step 600 (`0.8925`). During answer-only retention,
it recovers to `0.8925` exact-grid calc and `0.8750` sampled normal, beating
the hard-boundary branch at the final retention checkpoint.

The controls show the retained normal behavior remains calculator-mediated:
injection-zero and forced-random are near chance while oracle remains perfect.

This strengthens the case that local-target propagation can create a natural
result-level calculator interface that answer loss can retain. It does not yet
solve the project goal because every target-training update still scores all
forced result classes, so the method remains prescriptive/costly in the same
important way as hard assignment.

## Next

Do not rerun this same seed-2, 800+200, `hard_boundary` vs
`policy_reweighted_t1` comparison as novelty.

Allowed next tests:

- Seed replication of the longer `policy_reweighted_t1` retention result if
  stability is the question.
- A sampled/top-k/learned approximation to `policy_reweighted_t1` that reduces
  or avoids full forced-result enumeration.
- A schedule/stability diagnostic for the nonmonotonic target-training curve,
  but only if it changes the approximation or retention plan.

## Validation

```bash
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --operand-max 3 --steps 2 --retention-steps 2 --eval-every 1 --retention-eval-every 1 --control-eval-every 2 --eval-samples 32 --branches hard_boundary,policy_reweighted_t1 --output-root runs/2026-05-29_phase7_local_target_convergence_retention_gate/smoke_op3
python3 -m pytest tests/test_model.py -q
python3 researchMemory/scripts/generate_hypothesis_memories.py
python3 researchMemory/scripts/build_memory_index.py
python3 researchMemory/scripts/search_memory_fast.py "policy_reweighted_t1 answer-only retention exact-grid calc 0.8925" --top-k 5
```

`tests/test_model.py` passed with `114 passed`. The memory search returned the
new hypothesis memory as result 1.
