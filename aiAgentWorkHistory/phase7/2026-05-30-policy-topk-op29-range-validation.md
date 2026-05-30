# 2026-05-30 policy-topk op29 range validation

## Question

Does the lower-cost `topk8+unique24` assignment proposal preserve the op29
`rhead64` staged source/handoff ceiling, or was the op19 success range-limited?

This is a range and assignment-cost test. It keeps the same op29 product
decoder, hidden result head, late recovery schedule, and trusted 600-step
handoff gate that exact full-grid assignment already cleared, but reduces hard
improvement-assignment scoring from `59/59` result classes per prompt to
`24/59`.

## Comparator

Exact full-grid op29 `rhead64` effective seed `29`:

- source final `898/900 = 0.9978`
- source step `630` normal/source calc `0.9978`
- trusted handoff final `900/900 = 1.0000`
- handoff step `600` normal `1.0000`, learned calc `0.9967`,
  injection-zero `0.0244`, forced-random `0.0156`

## Runs

Sparse-assignment source:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op29_rhead64_topk8_unique24_source630_cpu/2026-05-30_113818_335991_model-c-op0-29-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed29
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op29_rhead64_topk8_unique24_handoff600_from_step630_cpu/2026-05-30_121551_265986_model-c-op0-29-fullgrid-adec-product/model-c-2digit-seed29
```

Both runs used CLI seed `27`, recorded as effective model seed `29`.

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
| `480` | `0.8944` | `0.0256` | `1.0000` | `0.0067` |
| `510` | `0.7322` | `0.0267` | `1.0000` | `0.0089` |
| `540` | `0.8878` | `0.0256` | `1.0000` | `0.0078` |
| `570` | `0.9511` | `0.0278` | `1.0000` | `0.0144` |
| `600` | `0.9289` | `0.0233` | `1.0000` | `0.0144` |
| `630` | `1.0000` | `0.0233` | `1.0000` | `0.0144` |

Final source eval was `900/900 = 1.0000`; the 128-sample final metrics
reported source exact / learned calc `1.0000`, injection-zero `0.0391`, and
forced-random `0.0078125`.

Trusted handoff snapshots:

| Step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9967` | `0.0300` | `0.9967` | `1.0000` | `0.0278` |
| `200` | `1.0000` | `0.0289` | `1.0000` | `1.0000` | `0.0133` |
| `300` | `1.0000` | `0.0367` | `1.0000` | `1.0000` | `0.0211` |
| `400` | `1.0000` | `0.0333` | `1.0000` | `1.0000` | `0.0133` |
| `500` | `1.0000` | `0.0444` | `1.0000` | `1.0000` | `0.0222` |
| `600` | `1.0000` | `0.0356` | `1.0000` | `1.0000` | `0.0189` |

Final handoff eval was `900/900 = 1.0000`; the 128-sample final metrics
reported normal / learned calc `1.0000`, injection-zero `0.015625`, and
forced-random `0.0078125`.

## Decision

```text
policy_topk_unique24_op29_range_handoff_positive
```

Interpretation:

- The policy-aware sparse assignment proposal preserved the op29 exact-grid
  source and trusted-handoff ceiling on the matched effective-seed-29
  comparator while scoring only `24/59` result classes.
- This is stronger scalability evidence than the op19 replications because the
  result vocabulary grew from `39` to `59` and the scored fraction fell from
  about `62%` to about `41%`.
- It is still not final proof of the thesis: the recipe uses hard improvement
  assignment, true-result forced-margin source shaping, a pretrained product
  decoder, hidden result-head capacity, and frozen-policy additive transfer.

Do not rerun the same effective-seed-29 op29 `rhead64` topk8+unique24
source630 plus handoff600 path as novelty. Next validation should use a fresh
op29 seed, op39/many-calculator cost with an explicit compute hypothesis, or a
less-prescriptive/non-enumerative replacement.
