# 2026-05-29 Replay-Memory Lower Fresh-Budget Gate

## Question

How far can replay-memory local targets reduce fresh forced-result scoring
while preserving the `policy_reweighted_t1` learning signal?

The prior `memory_policy_reweighted_t1_u8_m24` result was positive but still
used `8` fresh scores per prompt per step. This gate tests whether the memory
mechanism remains useful at lower fresh-score budgets.

## Runs

200-step budget sweep:

```text
runs/2026-05-29_phase7_memory_local_target_gate/lower_fresh_budget_200
```

800 target + 200 answer-only retention for the best lower-budget branch:

```text
runs/2026-05-29_phase7_memory_local_target_gate/memory_u2_m30_retention_800_200
```

## Results

### 200-Step Gate

| Branch | Fresh scored / step | Target width | Exact-grid calc | Sampled normal | True coverage | Target argmax | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | `32` | `32` | `0.3350` | `0.3438` | `0.8450` | `0.8350` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u8_m24` | `8` | `32` | `0.5900` | `0.5391` | `1.0000` | `0.9850` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u4_m28` | `4` | `32` | `0.5100` | `0.4844` | `1.0000` | `0.9850` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u2_m30` | `2` | `32` | `0.6025` | `0.6016` | `0.9925` | `0.9600` | `0.0234` | `0.0156` |
| `memory_policy_reweighted_t1_u1_m31` | `1` | `32` | `0.4075` | `0.4219` | `0.9250` | `0.8975` | `0.0234` | `0.0156` |

`u2_m30` was the best 200-step branch and used only `2` fresh forced-result
scores per prompt per step.

### 800+200 Retention Gate

| Branch | Target exact calc | Target sampled normal | Best sampled normal | Retention exact calc | Retention sampled normal | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u2_m30` | `0.9000` | `0.8750` | `0.8984` | `0.7850` | `0.7656` | `0.0234` | `0.0156` |

For comparison, the prior `u8_m24` retention gate reached target `0.9600`
exact calc / `0.9766` sampled normal and retained `0.8600` calc / `0.8750`
normal.

## Interpretation

Replay memory remains useful at a `16x` fresh-scoring reduction versus raw
uniform `u32`, and `u2_m30` slightly beats `u8_m24` in the 200-step gate.

The tradeoff is retention quality: `u2_m30` retains less strongly than
`u8_m24` after the 800+200 gate. The `u1_m31` branch is also below the useful
200-step budget floor.

## Next

Do not rerun this exact budget sweep or the same `u2_m30` retention gate as
novelty.

Useful next tests:

- stale-loss aging/rescoring, because cached losses become stale as the model
  changes;
- memory reset or finite memory windows;
- streaming/non-exhaustive prompt settings where per-prompt identity memory
  cannot eventually see all results;
- learned/generalized candidate memory that does not depend on fixed prompt
  identities.
