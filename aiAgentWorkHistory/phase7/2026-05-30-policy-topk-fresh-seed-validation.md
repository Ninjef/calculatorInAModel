# 2026-05-30 policy-topk fresh-seed validation

## Question

Does the lower-cost `topk8+unique24` assignment proposal survive a fresh seed
on the full staged op19 source-plus-handoff validation?

This tests whether the prior effective-seed-43 success was a one-seed artifact.
It keeps the assignment scoring cost at `24/39` result classes per prompt and
changes only the seed axis.

## Runs

Fresh-seed source:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_fresh_seed_source630_cpu/2026-05-30_112358_350343_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed47
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_fresh_seed_handoff600_from_step630_cpu/2026-05-30_113149_357254_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed47
```

Both runs used CLI seed `45`, recorded as effective model seed `47`.

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
| `540` | `0.9675` | `0.0375` | `1.0000` | `0.0250` |
| `570` | `0.9625` | `0.0250` | `1.0000` | `0.0225` |
| `600` | `0.9500` | `0.0175` | `1.0000` | `0.0200` |
| `630` | `1.0000` | `0.0325` | `1.0000` | `0.0250` |

Final source eval was `400/400 = 1.0000`.

Trusted handoff snapshots:

| Step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9475` | `0.0000` | `0.9475` | `1.0000` | `0.0225` |
| `200` | `0.9975` | `0.0100` | `0.9975` | `1.0000` | `0.0250` |
| `300` | `1.0000` | `0.0175` | `1.0000` | `1.0000` | `0.0250` |
| `400` | `1.0000` | `0.0075` | `1.0000` | `1.0000` | `0.0150` |
| `500` | `1.0000` | `0.0075` | `1.0000` | `1.0000` | `0.0250` |
| `600` | `1.0000` | `0.0475` | `1.0000` | `1.0000` | `0.0250` |

Final handoff eval was `400/400 = 1.0000`; final metrics reported
forced-random `0.03125`.

## Decision

```text
policy_topk_unique24_op19_fresh_seed_handoff_positive
```

Interpretation:

- The policy-aware assignment proposal is now a replicated op19 staged-transfer
  positive: effective seeds `43` and `47` both trained perfect sources and
  trusted additive handoffs while scoring `24/39` result classes.
- This strengthens the assignment-cost direction but does not solve the final
  thesis. The recipe still uses hard improvement assignment, forced-margin
  source shaping, a pretrained product decoder, and frozen-policy transfer.
- Do not run more op19 `rhead64` topk8+unique24 source630 plus trusted handoff
  fresh-seed replications as novelty. Next validation should move to
  range stress, many-calculator cost accounting, reduced prescriptiveness, or a
  non-enumerative replacement.
