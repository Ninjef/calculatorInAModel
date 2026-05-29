# 2026-05-29 - Adaptive Local-Target Proposal Gate

## Question

Can a loss-ranked adaptive proposal improve sparse `policy_reweighted_t1`
targets without scoring nearly every result class?

The previous sparse gate showed raw uniform/top-k candidate sets need near-full
coverage. This gate tests a different mechanism:

1. Sample a small no-replacement uniform seed set.
2. Score those forced-result candidates with answer loss.
3. Pick the lowest-loss sampled candidates.
4. Add integer neighborhoods around those low-loss candidates.
5. Build a policy-reweighted local target over the unique adaptive candidate
   set.

This is not oracle-assisted: expansion centers come from sampled answer loss,
not the true sum.

## Code

Extended:

```text
scripts/run_phase7_local_target_stage1_lift_gate.py
```

New branch family:

```text
adaptive_policy_reweighted_t1_u8_b4_r2
```

where `u` is initial uniform samples, `b` is the low-loss beam, and `r` is
integer expansion radius around each beam center.

The adaptive branch uses a dense unique-candidate target table so duplicate
expanded candidates do not receive duplicated probability mass.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_adaptive_local_target_gate/smoke_op3
```

Focused gate:

```text
runs/2026-05-29_phase7_adaptive_local_target_gate/adaptive_neighbor_200
```

Command:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches sampled_policy_reweighted_t1_k0_u32,adaptive_policy_reweighted_t1_u8_b4_r2,adaptive_policy_reweighted_t1_u8_b4_r3,adaptive_policy_reweighted_t1_u12_b4_r2 \
  --output-root runs/2026-05-29_phase7_adaptive_local_target_gate/adaptive_neighbor_200
```

## Results

Final metrics at 200 steps:

| Branch | Raw scored | Unique scored | Unique fraction | True coverage | Target argmax | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | 32 | 32.00 | n/a | 0.8450 | 0.8350 | 0.3350 | 0.3438 |
| `adaptive_policy_reweighted_t1_u8_b4_r2` | 28 | 18.42 | 0.4722 | 0.6350 | 0.6250 | 0.2025 | 0.2188 |
| `adaptive_policy_reweighted_t1_u8_b4_r3` | 36 | 22.08 | 0.5662 | 0.7425 | 0.7325 | 0.2600 | 0.2422 |
| `adaptive_policy_reweighted_t1_u12_b4_r2` | 32 | 20.74 | 0.5317 | 0.7700 | 0.7550 | 0.2700 | 0.2344 |

Curves:

| Branch | Step 0 calc | Step 50 | Step 100 | Step 150 | Step 200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | 0.0100 | 0.0400 | 0.1275 | 0.2600 | 0.3350 |
| `adaptive_policy_reweighted_t1_u8_b4_r2` | 0.0100 | 0.0525 | 0.1075 | 0.1750 | 0.2025 |
| `adaptive_policy_reweighted_t1_u8_b4_r3` | 0.0100 | 0.0550 | 0.1100 | 0.2300 | 0.2600 |
| `adaptive_policy_reweighted_t1_u12_b4_r2` | 0.0100 | 0.0525 | 0.1350 | 0.2150 | 0.2700 |

## Interpretation

This is negative for simple loss-neighborhood adaptive proposals.

Adaptive expansion improves true-result coverage relative to the initial small
seed set, but it clusters around low-loss sampled centers and wastes budget on
nearby duplicate/overlapping results. At similar raw scoring cost, it produces
fewer unique candidates and lower calculator learning than raw no-replacement
uniform `u32`.

The current evidence says the next approximation should not be a simple
loss-ranked neighborhood expansion. It needs either:

- a learned proposal that predicts useful candidates without clustering so
  heavily,
- an importance/bias correction that makes partial candidate coverage less
  damaging, or
- a different local-target construction that does not require high true-result
  coverage.

## Next

Do not repeat the same seed-2 200-step adaptive neighborhood gate over
`u8_b4_r2/u8_b4_r3/u12_b4_r2` as novelty.

Allowed next tests:

- Train or fit a candidate proposal distribution and evaluate candidate
  coverage/calc learning against this raw uniform and adaptive-neighborhood
  baseline.
- Try an importance-corrected sampled policy-reweighted target if it changes
  the bias/variance problem rather than merely adding samples.
- Pivot back to source-acquisition-for-handoff geometry if no better
  local-target proposal mechanism is available.

## Validation

```bash
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --operand-max 3 --steps 2 --eval-every 1 --control-eval-every 2 --eval-samples 32 --branches adaptive_policy_reweighted_t1_u4_b2_r1 --output-root runs/2026-05-29_phase7_adaptive_local_target_gate/smoke_op3
python3 -m pytest tests/test_model.py -q
python3 researchMemory/scripts/generate_hypothesis_memories.py
python3 researchMemory/scripts/build_memory_index.py
python3 researchMemory/scripts/search_memory_fast.py "adaptive loss neighborhood policy_reweighted sparse local target underperformed raw uniform" --top-k 5
```

`tests/test_model.py` passed with `114 passed`. The memory search returned the
new adaptive-neighborhood hypothesis memory as result 1.
