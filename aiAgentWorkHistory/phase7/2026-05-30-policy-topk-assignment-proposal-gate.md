# 2026-05-30 policy-topk assignment proposal gate

## Question

Can a non-uniform candidate proposal reduce hard improvement-assignment scoring
cost while preserving the op19 `rhead64` source signal?

Prior gates showed that duplicate-prone uniform sampling fails, fixed stale
exact targets fail, and duplicate-free uniform sampling helps but still misses
the exact ceiling. This gate tests a policy-aware proposal: reserve candidate
slots for the model's current result-policy top-k classes, then fill the rest
with unique random result classes.

## Implementation

Added:

- `--result-policy-improvement-assignment-policy-topk-count`
- priority candidate support in sampled hard-assignment candidate construction
- tests showing top-k candidates are kept first and remaining slots are unique

The proposal remains non-prescriptive: it uses the model's own result-policy
distribution, not the oracle answer/result, to choose priority candidates.

## Runs

All runs used the same op19 `rhead64`, product-decoder, one-negative
forced-margin source gate and CLI seed `41` / effective model seed `43`.

Existing comparators:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_exact_source200_cpu
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_unique16_source200_cpu
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_unique32_source200_cpu
```

Policy-aware proposal runs:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique16_source200_cpu/2026-05-30_110237_340020_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_source200_cpu/2026-05-30_110045_344562_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique32_source200_cpu/2026-05-30_105838_258797_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

## Results

| Assignment | Scored results | Step-200 true coverage | Step-200 target acc | Best snapshot normal/calc | Final eval | Approx wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.9900` | `0.8625` at step `150` | `294/400 = 0.7350` | `115.5s` |
| Unique16 | `16/39` | `0.6525` | `0.5317` | `0.3625` at step `200` | `162/400 = 0.4050` | `88.0s` |
| Topk8+unique16 | `16/39` | `1.0000` | `0.9333` | `0.6850` at step `150` | `269/400 = 0.6725` | `88.2s` |
| Topk8+unique24 | `24/39` | `1.0000` | `1.0000` | `0.7725` at step `200` | `300/400 = 0.7500` | `93.9s` |
| Unique32 | `32/39` | `0.9275` | `0.8156` | `0.6250` at step `200` | `244/400 = 0.6100` | `100.5s` |
| Topk8+unique32 | `32/39` | `1.0000` | `0.9412` | `0.7925` at step `150` | `344/400 = 0.8600` | `103.3s` |

## Decision

```text
policy_topk_unique_assignment_proposal_mixed_positive
```

Interpretation:

- Policy top-k proposal is the first assignment-cost reduction mechanism that
  preserves much of the exact source signal.
- The mechanism changes the failure mode: true-result coverage hits `1.0000`
  even at `16/39` scored classes, while duplicate-free uniform needed `32/39`
  to reach only `0.9275`.
- The result is not yet a solved scalable recipe. It is a single op19 source
  gate on one seed, not a trusted handoff, range, or many-calculator result.

Do not run more topk8 unique count ladders on the same op19 `rhead64` 200-step
gate as novelty. Next tests should validate policy-aware proposals in the
staged recipe: longer source plus handoff, fresh seed, larger operand range, or
many-calculator cost accounting.
