# 2026-05-30 policy-topk source handoff validation

## Question

Does the promising `topk8+unique24` assignment proposal survive beyond the
200-step source screen and produce a usable staged additive handoff?

This validates the policy-aware proposal where it matters: longer bottleneck
source training with late recovery, followed by a trusted frozen-policy
non-bottleneck handoff.

## Runs

Longer source:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_source630_cpu/2026-05-30_110914_368147_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_handoff600_from_step630_cpu/2026-05-30_111626_687062_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed43
```

Both runs used CLI seed `41`, recorded as effective model seed `43`.

Source configuration:

- op19 exhaustive grid, `rhead64`, product decoder
- one-negative forced-margin source objective
- late recovery at step `600`
- hard-improvement assignment candidate scoring reduced to `24/39` result
  classes via `topk8+unique24`

## Results

Source snapshots:

| Step | Normal / calc | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: |
| `540` | `0.9625` | `0.0150` | `1.0000` | `0.0100` |
| `570` | `0.9300` | `0.0225` | `1.0000` | `0.0100` |
| `600` | `0.9400` | `0.0175` | `1.0000` | `0.0300` |
| `630` | `1.0000` | `0.0275` | `1.0000` | `0.0300` |

Final source eval was `400/400 = 1.0000`.

Trusted handoff snapshots:

| Step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9900` | `0.0175` | `0.9900` | `0.9900` | not primary |
| `200` | `1.0000` | `0.0100` | `1.0000` | `1.0000` | not primary |
| `400` | `1.0000` | `0.0250` | `1.0000` | `1.0000` | `0.0175` |
| `500` | `1.0000` | `0.0425` | `1.0000` | `1.0000` | `0.0225` |
| `600` | `1.0000` | `0.0200` | `1.0000` | `1.0000` | `0.0325` |

Final handoff eval was `400/400 = 1.0000`.

## Decision

```text
policy_topk_unique24_op19_source_handoff_positive
```

Interpretation:

- The policy-aware assignment proposal survives the first real staged-transfer
  validation: it trains a perfect op19 source and transfers into additive
  non-bottleneck mode with low calculator-ablation controls.
- This is a meaningful scalability improvement over exact assignment within
  this gate because assignment scoring drops from `39` result classes to `24`
  per prompt while preserving source and handoff quality.
- It is not final proof of scalability or non-prescriptiveness. The method
  still uses hard improvement assignment, forced-margin pressure, a pretrained
  semantic decoder, and frozen-policy transfer.

Do not rerun the same effective-seed-43 op19 `rhead64` topk8+unique24
source630 plus handoff600 path as novelty. Next validation should use a fresh
seed, larger operand range, many-calculator cost accounting, or reduced
prescriptiveness.
