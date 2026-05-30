# 2026-05-30 policy-topk op29 fresh-seed validation

## Question

Does the `topk8+unique24` op29 sparse-assignment source/handoff result
replicate on the existing exact full-grid fresh-range seed?

This repeats the op29 `rhead64` staged recipe on CLI seed `31` / effective seed
`33`, matching the exact full-grid comparator seed. The assignment scoring cost
remains `24/59` result classes per prompt instead of exact `59/59`.

## Comparator

Exact full-grid op29 `rhead64` effective seed `33`:

- source final `897/900 = 0.9967`
- source step `630` normal/source calc `0.9967`
- trusted handoff final `900/900 = 1.0000`
- handoff step `600` normal `1.0000`, learned calc `1.0000`,
  injection-zero `0.0344`, forced-random `0.0111`

## Runs

Sparse-assignment source:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op29_rhead64_topk8_unique24_fresh_seed_source630_cpu/2026-05-30_122251_533761_model-c-op0-29-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed33
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op29_rhead64_topk8_unique24_fresh_seed_handoff600_from_step630_cpu/2026-05-30_123756_313224_model-c-op0-29-fullgrid-adec-product/model-c-2digit-seed33
```

Both runs used CLI seed `31`, recorded as effective model seed `33`.

Source configuration:

- op29 exhaustive grid, `rhead64`, product decoder
- one-negative forced-margin source objective
- late recovery at step `600`
- hard-improvement assignment candidate scoring reduced to `24/59` result
  classes via `topk8+unique24`

## Results

Source snapshots:

| Step | Normal / calc | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: |
| `480` | `0.8511` | `0.0233` | `1.0000` | `0.0222` |
| `510` | `0.9400` | `0.0244` | `1.0000` | `0.0133` |
| `540` | `0.8544` | `0.0267` | `1.0000` | `0.0211` |
| `570` | `0.9033` | `0.0256` | `1.0000` | `0.0167` |
| `600` | `0.9211` | `0.0311` | `1.0000` | `0.0089` |
| `630` | `0.9989` | `0.0200` | `1.0000` | `0.0133` |

Final source eval was `899/900 = 0.9989`; the 128-sample final metrics
reported source exact / learned calc `1.0000`, injection-zero `0.03125`, and
forced-random `0.015625`.

Trusted handoff snapshots:

| Step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9900` | `0.0278` | `0.9911` | `0.9989` | `0.0156` |
| `200` | `1.0000` | `0.0278` | `1.0000` | `0.9989` | `0.0133` |
| `300` | `1.0000` | `0.0400` | `1.0000` | `1.0000` | `0.0156` |
| `400` | `1.0000` | `0.0344` | `1.0000` | `0.9989` | `0.0178` |
| `500` | `1.0000` | `0.0244` | `1.0000` | `0.9978` | `0.0189` |
| `600` | `1.0000` | `0.0333` | `1.0000` | `0.9989` | `0.0111` |

Final handoff eval was `900/900 = 1.0000`; the 128-sample final metrics
reported normal / learned calc `1.0000`, injection-zero `0.03125`, and
forced-random `0.015625`.

## Decision

```text
policy_topk_unique24_op29_range_replicates
```

Interpretation:

- The policy-aware sparse assignment proposal now has replicated op29 range
  evidence. Effective seeds `29` and `33` both preserved source and trusted
  handoff quality while scoring only `24/59` result classes.
- This strengthens the scalable-assignment story, but it still does not solve
  the thesis: the recipe uses hard improvement assignment, true-result
  forced-margin source shaping, a pretrained product decoder, hidden result
  head capacity, and frozen-policy additive transfer.
- Do not run more op29 `rhead64` topk8+unique24 source630 plus trusted handoff
  seed replications as novelty. Next validation should move to many-calculator
  cost/accounting, op39 with an explicit compute hypothesis, or a
  less-prescriptive/non-enumerative replacement.
