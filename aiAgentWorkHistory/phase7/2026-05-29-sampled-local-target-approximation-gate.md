# 2026-05-29 - Sampled Local-Target Approximation Gate

## Question

Can the `policy_reweighted_t1` local-target signal be approximated without
scoring every forced calculator result class on every update?

This is the scalability follow-up to the exact-grid local-target gates. The
target construction is still answer-derived and non-oracle, but the candidate
set is sparse:

```text
sampled_policy_reweighted_t1_k<TOP_K>_u<UNIFORM_SAMPLES>
```

For each example, the branch scores only the union of current-policy top-k
result classes and no-replacement uniform result samples, then builds the same
policy-reweighted target over that candidate set.

## Code

Extended:

```text
scripts/run_phase7_local_target_stage1_lift_gate.py
```

New branch family:

```text
sampled_policy_reweighted_t1_k8_u8
sampled_policy_reweighted_t1_k0_u24
...
```

Logged metrics include candidate coverage, unique scored results, scored
fraction, target true probability, and candidate-set target argmax accuracy.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_sampled_local_target_gate/smoke_op3
runs/2026-05-29_phase7_sampled_local_target_gate/smoke_no_replacement_op3
```

Main sparse gate:

```text
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_no_replacement_200
```

Near-full diagnostic:

```text
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_near_full_200
```

Exact sanity check:

```text
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_exact_alone_200
```

Superseded with-replacement exploratory runs are under:

```text
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_sampled_200
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_sampled_coverage_ladder_200
runs/2026-05-29_phase7_sampled_local_target_gate/policy_t1_uniform_only_200
```

They motivated switching uniform candidates to no-replacement sampling and are
not the main evidence.

## Results

Exact `policy_reweighted_t1` sanity check:

| Branch | Scored fraction | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: |
| `policy_reweighted_t1` | 1.0000 | 0.5600 | 0.5391 |

No-replacement sparse candidate gate:

| Branch | Unique scored | Scored fraction | True coverage | Target argmax | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k8_u8` | 14.36 | 0.4103 | 0.5075 | 0.3875 | 0.0925 | 0.0859 |
| `sampled_policy_reweighted_t1_k0_u16` | 16.00 | 0.4103 | 0.4050 | 0.4050 | 0.1975 | 0.1797 |
| `sampled_policy_reweighted_t1_k0_u24` | 24.00 | 0.6154 | 0.6250 | 0.6200 | 0.2800 | 0.2734 |
| `sampled_policy_reweighted_t1_k0_u32` | 32.00 | 0.8205 | 0.8450 | 0.8350 | 0.3350 | 0.3438 |

Near-full diagnostic:

| Branch | Unique scored | Scored fraction | True coverage | Target argmax | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u36` | 36.00 | 0.9231 | 0.9300 | 0.9200 | 0.4100 | 0.3906 |
| `sampled_policy_reweighted_t1_k0_u39` | 39.00 | 1.0000 | 1.0000 | 0.9925 | 0.6250 | 0.5547 |

The `u39` branch scores the full result vocabulary through the sparse path. It
does not exactly match the exact branch, likely because per-example random
candidate ordering changes floating-point accumulation and the optimization
trajectory, but it confirms the sparse-path implementation can carry the
full-enum signal.

## Interpretation

This is mixed-negative for naive sparse approximation.

Sparse candidate scoring is directionally useful, but it does not preserve the
exact `policy_reweighted_t1` learning signal at small or moderate candidate
counts. Current-policy top-k plus uniform candidates is worse than uniform-only
sampling at the same broad budget, suggesting early top-k candidates anchor the
target to wrong high-probability results. Uniform no-replacement sampling
improves smoothly with coverage, but even scoring `32/39` result classes only
reaches `0.3350` exact-grid calculator accuracy at 200 steps.

The practical lesson is that naive sparse sampling is not yet the scalable
credit-assignment method. It needs a better proposal mechanism, learned
candidate generator, variance/bias correction, or a schedule that raises true
candidate coverage without approaching full enumeration.

## Next

Do not repeat the same seed-2 200-step sparse ladder over
`k8_u8/k0_u16/k0_u24/k0_u32/k0_u36/k0_u39` as novelty.

Allowed next tests:

- A smarter proposal distribution that improves true-result candidate coverage
  without scoring nearly all result classes.
- A learned/top-k-after-warmup candidate generator evaluated against the exact
  `policy_reweighted_t1` ceiling.
- An importance-corrected sampled target if it changes the bias/variance story,
  not merely another raw uniform sample count.

## Validation

```bash
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --operand-max 3 --steps 2 --eval-every 1 --control-eval-every 2 --eval-samples 32 --branches sampled_policy_reweighted_t1_k0_u4 --output-root runs/2026-05-29_phase7_sampled_local_target_gate/smoke_no_replacement_op3
python3 -m pytest tests/test_model.py -q
python3 researchMemory/scripts/generate_hypothesis_memories.py
python3 researchMemory/scripts/build_memory_index.py
python3 researchMemory/scripts/search_memory_fast.py "naive sparse sampled policy_reweighted_t1 candidate coverage near full enumeration" --top-k 5
```

`tests/test_model.py` passed with `114 passed`. The memory search returned the
new sparse-target hypothesis memory as result 1.
