# 2026-05-29 Replay-Memory Rescore Gate

## Question

Is stale cached loss the reason low-fresh replay memory retains worse, and can
simple top-cached-candidate rescoring fix it?

The prior best low-fresh branch, `memory_policy_reweighted_t1_u2_m30`, used
only `2` fresh forced-result scores per prompt per step but retained worse than
`u8_m24`. This test refreshes selected cached candidate losses before target
construction.

## Code Change

Added optional `_rN` syntax to replay-memory local-target branches:

```text
memory_policy_reweighted_t1_u2_m30_r4
```

The suffix rescores the top `N` cached candidates with the current model each
step, updates the memory table, and records:

- `target_rescored_results`
- `target_forced_scores_per_step`

Existing branch names without `_rN` keep the prior behavior.

## Runs

Smoke:

```text
runs/2026-05-29_phase7_memory_local_target_gate/smoke_rescore_op3
```

200-step comparison:

```text
runs/2026-05-29_phase7_memory_local_target_gate/rescore_budget_200
```

800 target + 200 answer-only retention:

```text
runs/2026-05-29_phase7_memory_local_target_gate/memory_u2_m30_r2_retention_800_200
```

## Results

### 200-Step Gate

| Branch | Forced scores / step | Exact-grid calc | Sampled normal | True coverage | Target argmax | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u2_m30` | `2` | `0.6025` | `0.6016` | `0.9925` | `0.9600` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u2_m30_r2` | `4` | `0.6025` | `0.6016` | `0.9925` | `0.9600` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u2_m30_r4` | `6` | `0.5300` | `0.5781` | `0.9925` | `0.9600` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u2_m30_r8` | `10` | `0.4675` | `0.4609` | `0.9925` | `0.9625` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u8_m24` | `8` | `0.5900` | `0.5391` | `1.0000` | `0.9850` | `0.0234` | `0.0156` |

### 800+200 Retention Gate

| Branch | Forced scores / step | Target exact calc | Target sampled normal | Retention exact calc | Retention sampled normal | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u2_m30_r2` | `4` | `0.9000` | `0.8750` | `0.7850` | `0.7656` | `0.0234` | `0.0156` |

This exactly tied the prior no-rescore `u2_m30` 800+200 gate.

## Interpretation

Simple top-cached-candidate rescoring is not the retention fix.

- `r2` ties no-rescore but doubles forced-score cost.
- `r4` and `r8` hurt the short gate.
- The likely next bottleneck is the transductive memory setup itself, not just
  stale loss values inside the memory.

## Next

Do not repeat this rescore-count sweep as novelty.

Useful next tests:

- finite/reset memory;
- streaming or non-exhaustive prompts where per-prompt identity memory cannot
  eventually see all results;
- learned/generalized candidate memory.
