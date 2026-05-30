# 2026-05-30 op29 Hidden Result-Head Capacity Diagnostic

## Question

Was the op29 range-stress failure caused by insufficient source policy
capacity in the shallow result-space head?

This is a materially different source-capacity diagnostic. It keeps the same
op29 product oracle decoder and the same automated one-negative forced-margin
source schedule, but changes the result-space policy head from a linear
projection to a hidden MLP head:

```text
--calculator-result-head-hidden-size 64
```

## Runs

Source:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_rhead64_source630_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_rhead64_handoff600_from_step630_cpu
```

The run used the same op29 product oracle semantic decoder as the prior range
stress. CLI seed was `27`; run directories record effective model seed `29`.

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `390` | `0.9022` | `0.9022` | `0.0222` | `1.0000` | `0.0122` |
| `450` | `0.9211` | `0.9211` | `0.0200` | `1.0000` | `0.0167` |
| `540` | `0.9411` | `0.9411` | `0.0256` | `1.0000` | `0.0078` |
| `570` | `0.9467` | `0.9467` | `0.0278` | `1.0000` | `0.0144` |
| `600` | `0.8767` | `0.8767` | `0.0233` | `1.0000` | `0.0144` |
| `630` | `0.9978` | `0.9978` | `0.0233` | `1.0000` | `0.0144` |

Final source eval was `898/900 = 0.9978`. The 128-sample diagnostic summary
reported source exact match / learned calc `0.9922`, injection-zero `0.0391`,
and forced-random `0.0078`.

Capacity/cost:

| Source | Total params | Result-head params |
| --- | ---: | ---: |
| op29 shallow result head | `28,311` | `7,611` |
| op29 `rhead64` | `32,791` | `12,091` |

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9833` | `0.0500` | `0.9844` | `0.9989` | `0.0311` |
| `200` | `0.9989` | `0.0389` | `1.0000` | `0.9989` | `0.0111` |
| `300` | `0.9989` | `0.0478` | `1.0000` | `0.9989` | `0.0256` |
| `400` | `1.0000` | `0.0311` | `1.0000` | `0.9978` | `0.0156` |
| `500` | `1.0000` | `0.0456` | `1.0000` | `1.0000` | `0.0211` |
| `600` | `1.0000` | `0.0244` | `1.0000` | `0.9967` | `0.0156` |

Final handoff eval was `900/900 = 1.0000`. The 128-sample diagnostic summary
reported normal `1.0000`, injection-zero `0.0156`, forced-random `0.0078`,
and learned calculator accuracy `0.9922`.

## Decision

```text
op29_hidden_result_head_capacity_positive_but_prescriptive
```

Interpretation:

- The op29 range failure was strongly source-capacity sensitive. Adding a
  hidden result head rescued source acquisition without the extra low-LR
  continuation ladder and produced a perfect trusted handoff.
- The calculator path is causal: zero-injection and forced-random controls stay
  low while normal/oracle are perfect.
- This improves the staged benchmark's range story, but it is still not the
  final scalable/non-prescriptive method. Training remains full-grid hard
  assignment plus true-result forced-margin pressure, and the hidden head adds
  per-calculator/source-head parameters (`+4,480` result-head params at op29).

Do not rerun this same op29 `rhead64`, effective-seed-29 source-plus-handoff as
novelty. Further range/capacity work should test whether the hidden-head result
scales to fresh seeds, larger ranges, or cheaper assignment, or should remove
the hard assignment / true-result forcing that still makes this prescriptive.
