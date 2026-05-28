# 2026-05-28 Result-Policy Soft Diversity Gate

## Question

Can a non-prescriptive entropy/batch-diversity constraint prevent refreshed
online-shadow Stage 1 from collapsing to one calculator result?

## Implementation

- Added `--result-policy-entropy-weight`.
- Added `--result-policy-batch-diversity-weight`.
- Added `--result-policy-stabilization-temperature`.
- Added `--result-policy-stabilization-decay-steps`.
- Added training-curve metrics for soft entropy/effective results,
  batch-marginal diversity, hard-marginal diversity, and argmax/top-3 result
  accuracy.

## Runs

Run root:

```text
runs/2026-05-28_phase7_shadow_refresh_result_policy_diversity_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- h32 validation-gradient online shadow module.
- `shadow_feedback_weight=1.0`.
- refresh every `50` steps.
- 200 steps, snapshots every `25`.

Results:

| Entropy | Diversity | Apply clamp | Final exact | Best snapshot | Final hard effective results | Final soft marginal effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | `1.0` | none | `0.015` | `0.0475` | `1.00` | `1.00` |
| `0.01` | `1.0` | `10` | `0.005` | `0.0400` | `1.00` | `1.00` |
| `0.0` | `100.0` | `10` | `0.070` | `0.0800` | `9.14` | `35.11` |

## Conclusion

```text
result_policy_soft_diversity_stabilization_stage1_negative
```

The soft diversity objective can mechanically broaden usage when scaled very
high, but broader usage alone does not make the calculator requests useful.
The next useful branch needs to connect diversity to per-example improvement,
not merely maximize the marginal result entropy.

## Anti-Regression Note

Do not repeat low soft diversity weight `1.0` with or without clamp `10`, or
high diversity weight `100` with clamp `10`, on this same refreshed h32
validation-gradient setup as novelty.
