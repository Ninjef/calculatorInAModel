# 2026-05-30 op29 rhead64 Fresh-Seed Replication

## Question

Does the hidden result-head capacity fix for the op29 forced-margin range
stress survive a fresh seed, or was the first perfect handoff a lucky source
run?

This repeats the op29 `rhead64` source-plus-handoff with a new CLI seed `31`
(effective model seed `33`). It keeps the same op29 product oracle decoder,
automated one-negative forced-margin schedule, and trusted 600-step
frozen-policy additive handoff gate.

## Runs

Source:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_rhead64_fresh_seed_source630_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_rhead64_fresh_seed_handoff600_from_step630_cpu
```

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `510` | `0.8611` | `0.8611` | `0.0244` | `1.0000` | `0.0233` |
| `540` | `0.8956` | `0.8956` | `0.0267` | `1.0000` | `0.0167` |
| `570` | `0.8378` | `0.8378` | `0.0256` | `1.0000` | `0.0167` |
| `600` | `0.7122` | `0.7122` | `0.0311` | `1.0000` | `0.0089` |
| `630` | `0.9967` | `0.9967` | `0.0200` | `1.0000` | `0.0133` |

Final source eval was `897/900 = 0.9967`. The 128-sample diagnostic summary
reported source exact match / learned calc `0.9922`.

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9967` | `0.0500` | `1.0000` | `0.9978` | `0.0222` |
| `200` | `0.9978` | `0.0522` | `1.0000` | `0.9978` | `0.0222` |
| `300` | `1.0000` | `0.0456` | `1.0000` | `0.9989` | `0.0200` |
| `400` | `1.0000` | `0.0300` | `1.0000` | `1.0000` | `0.0222` |
| `500` | `1.0000` | `0.0333` | `1.0000` | `1.0000` | `0.0200` |
| `600` | `1.0000` | `0.0344` | `1.0000` | `1.0000` | `0.0111` |

Final handoff eval was `900/900 = 1.0000`. The 128-sample diagnostic summary
reported normal `1.0000` and learned calculator accuracy `0.9922`.

## Decision

```text
op29_hidden_result_head_capacity_replicates_but_prescriptive
```

Interpretation:

- The op29 `rhead64` fix is no longer a one-seed capacity rescue. A fresh seed
  repeated near-perfect source acquisition and perfect trusted additive
  handoff with low zero-injection and forced-random controls.
- The source trajectory still shows late-window instability: step `600`
  dropped to `0.7122` before recovering to `0.9967` at step `630`. The late
  recovery window remains important.
- This strengthens the staged range story but does not solve the goal. The
  successful recipe still uses full-grid hard assignment, true-result
  forced-margin pressure, frozen-policy transfer, and extra result-head
  capacity per calculator.

Do not rerun the completed op29 `rhead64` effective-seed-29 or effective-seed-33
source-plus-handoff pairs as novelty. Next range work should stress a new axis
such as larger operand range or many-calculator cost, or reduce/remove the
prescriptive full-grid assignment pressure.
