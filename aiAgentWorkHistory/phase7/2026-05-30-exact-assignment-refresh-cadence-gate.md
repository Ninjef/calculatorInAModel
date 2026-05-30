# 2026-05-30 exact assignment refresh cadence gate

## Question

Can hard improvement-assignment cost be reduced by refreshing exact full-result
targets only every N steps and reusing cached targets between refreshes?

This is a different cost-reduction mechanism from uniform sampled candidates:
it preserves full result coverage when targets are refreshed, then tests whether
those exact targets stay useful long enough to amortize scoring.

## Implementation

Added:

- `--result-policy-improvement-assignment-refresh-interval`
- cached exact assignment targets for fixed `--exhaustive-grid-batch` runs
- validation that cached refresh requires exact assignment and fixed prompts
- CSV metrics for refresh interval, refreshed flag, and target age

Cached steps set `result_policy_improvement_assignment_scored_count=0`, while
refreshed steps score the full result vocabulary.

## Runs

All runs used the same op19 `rhead64`, product-decoder, one-negative
forced-margin source gate and CLI seed `41` / effective model seed `43`.

Existing exact ceiling:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_exact_source200_cpu/2026-05-30_102233_252420_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

Refresh cadence runs:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_refresh2_source200_cpu/2026-05-30_104104_498374_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_refresh5_source200_cpu/2026-05-30_103902_043039_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

## Results

| Assignment cadence | Full refreshes over 201 steps | Best snapshot normal/calc | Step-200 target acc | Final eval | Approx wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact every step | `201` | `0.8625` at step `150` | `0.9900` | `294/400 = 0.7350` | `115.5s` |
| Refresh every 2 | `101` | `0.5875` at step `200` | `0.9603` | `237/400 = 0.5925` | `106.4s` |
| Refresh every 5 | `41` | `0.4950` at step `200` | `0.8600` | `198/400 = 0.4950` | `105.1s` |

Snapshot controls stayed low:

| Cadence | Step-200 normal | Injection-zero | Oracle | Forced-random | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | `0.7375` | `0.0200` | `1.0000` | `0.0200` | `0.7375` |
| Refresh every 2 | `0.5875` | `0.0200` | `1.0000` | `0.0200` | `0.5875` |
| Refresh every 5 | `0.4950` | `0.0200` | `1.0000` | `0.0200` | `0.4950` |

## Decision

```text
fixed_cadence_exact_assignment_refresh_mixed_negative
```

Interpretation:

- Temporal amortization is less destructive than uniform sampled candidates,
  but it still does not preserve the exact source signal.
- Refresh2 kept decent step-200 target accuracy (`0.9603`) yet learned much
  more slowly than exact, suggesting that target freshness matters for the
  policy dynamics even when the cached target is often still correct.
- The full diagnostic wall-clock gain was small because snapshots,
  checkpoints, forced-margin scoring, and ordinary training still dominate
  local runtime at this scale.

Do not run more fixed refresh-interval ladders on this op19 `rhead64` gate as
novelty. Further temporal cost reduction needs an adaptive freshness/trust
criterion or predictive target update, and it must show both source retention
and real compute savings.
