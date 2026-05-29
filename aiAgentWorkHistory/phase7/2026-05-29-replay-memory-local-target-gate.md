# 2026-05-29 Replay-Memory Local-Target Gate

## Question

Can the `policy_reweighted_t1` local-target signal be approximated by scoring
only a few fresh result candidates per step and reusing previously observed
forced-result losses?

This is a new mechanism relative to the failed sparse proposal ladders: the
candidate set is not just top-k/uniform or a hand-coded low-loss neighborhood.
It amortizes candidate scoring over time with a per-prompt replay memory.

## Code Change

Added `memory_policy_reweighted_t<T>_u<U>_m<M>` branches to
`scripts/run_phase7_local_target_stage1_lift_gate.py`.

For each prompt, the runner now maintains a table of observed forced-result
losses. Each target-training step:

1. samples and scores `U` fresh uniform result classes;
2. updates the per-prompt loss memory;
3. builds the policy-reweighted target from the fresh candidates plus the best
   `M` cached candidates.

The default tested branch was `memory_policy_reweighted_t1_u8_m24`, so the
target has width `32` but only `8` fresh forced-result scores per step.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_memory_local_target_gate/smoke_op3
```

200-step comparison:

```text
runs/2026-05-29_phase7_memory_local_target_gate/memory_u8_m24_vs_uniform_u32_200
```

800 target + 200 answer-only retention:

```text
runs/2026-05-29_phase7_memory_local_target_gate/memory_u8_m24_retention_800_200
```

## Results

### 200-Step Gate

| Branch | Fresh scored / step | Target width | Exact-grid calc | Sampled normal | True coverage | Target argmax | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | `32` | `32` | `0.3350` | `0.3438` | `0.8450` | `0.8350` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u8_m24` | `8` | `32` | `0.5900` | `0.5391` | `1.0000` | `0.9850` | `0.0234` | `0.0156` |

The memory branch had observed all `39` result classes per prompt by step 200,
but it did so through repeated `8`-fresh-candidate updates instead of 32 fresh
scores every step.

### 800+200 Retention Gate

| Branch | Target exact calc | Target sampled normal | Retention exact calc | Retention sampled normal | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u8_m24` | `0.9600` | `0.9766` | `0.8600` | `0.8750` | `0.0234` | `0.0156` |

For comparison, the earlier exact full-enum `policy_reweighted_t1` 800+200
gate finished retention at `0.8925` exact calc and `0.8750` sampled normal.

## Interpretation

Replay memory is the first sparse local-target approximation to beat raw
uniform `u32` decisively while scoring fewer fresh result classes per step.
It also preserves answer-only retention behavior in the same sampled-normal
range as the exact branch.

The caveat is important: this is transductive on the fixed exhaustive grid.
The memory eventually observes all result classes for each prompt, so this is
not yet a scalable proof for fresh prompts or large data streams.

## Next

Do not rerun this exact `u8_m24` comparison as novelty.

Useful next tests:

- lower the fresh scoring budget (`u4` or below);
- add aging/rescoring to handle stale losses as the model changes;
- test whether a learned/generalized proposal can replace per-prompt identity
  memory.

## Verification

```text
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --operand-max 3 --steps 2 --eval-every 1 --control-eval-every 2 --eval-samples 32 --branches memory_policy_reweighted_t1_u2_m2,sampled_policy_reweighted_t1_k0_u4 --output-root runs/2026-05-29_phase7_memory_local_target_gate/smoke_op3
```
