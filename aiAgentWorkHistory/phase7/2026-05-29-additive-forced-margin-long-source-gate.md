# 2026-05-29 Additive Forced-Margin Long Source Gate

## Question

Does the one-negative forced-margin handoff lift compound at longer full-grid
source horizons, and can it beat the scheduled forced-true step-600 handoff?

The previous 200-step one-negative source gate reached `0.6600` trusted
handoff final eval, beating matched 200-step scheduled forced-true (`0.4150`)
but not the longer scheduled forced-true step-600 source (`0.7725`).

## Runs

Fresh 600-step source:

```text
runs/2026-05-29_phase7_additive_forced_margin/op19_long_neg1
```

Continuation from the exact prior positive 200-step checkpoint:

```text
runs/2026-05-29_phase7_additive_forced_margin/op19_continue_from_step200_neg1
```

Trusted handoffs:

```text
runs/2026-05-29_phase7_additive_forced_margin/op19_long_neg1/handoff600_step400
runs/2026-05-29_phase7_additive_forced_margin/op19_long_neg1/handoff600_step600
runs/2026-05-29_phase7_additive_forced_margin/op19_continue_from_step200_neg1/handoff600_step200
```

## Setup

Common source setup:

- `operand_max=19`, exhaustive grid.
- Seed `13`.
- no-decay source stabilization:
  - result-policy improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`
- one-negative forced-margin:
  - weight `0.5`
  - margin `0.05`
  - start step `50` for the fresh source
  - start step `0` for continuation from the already-active step-200 source

Trusted handoff setup:

- `calculator_bottleneck_mode=none`
- `--freeze-calculator-policy`
- 600 downstream answer-loss steps.

## Fresh 600-Step Source

Source trajectory:

| Step | Source calc | Snapshot normal | Margin loss | Source final eval |
| ---: | ---: | ---: | ---: | ---: |
| `200` | `0.3025` | `0.3100` | `0.0026` | n/a |
| `400` | `0.4525` | `0.4600` | `0.0000` | n/a |
| `600` | `0.5225` | `0.4975` | `0.0000` | `0.4800` |

Geometry:

| Source checkpoint | Source calc | Forced best=true | True-best gap | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: |
| step `200` | `0.3025` | `0.7225` | `0.0099` | `1.3404` |
| step `400` | `0.4525` | `0.9450` | `0.0011` | `1.2446` |
| step `600` | `0.5225` | `0.9925` | `0.0000` | `1.2365` |

Trusted handoff:

| Source checkpoint | Final eval | Step-600 normal | Injection-zero | Forced-random | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `400` | `0.7030` | `0.7025` | `0.0000` | `0.0400` | `0.4250` |
| step `600` | `0.7330` | `0.7500` | `0.0000` | `0.0225` | `0.4975` |

## Continuation From Prior Step-200 Positive

The fresh 600-step source does not exactly extend the previous 200-step result,
because the sampled-negative objective is sensitive to RNG path and the fresh
run used a different snapshot/checkpoint cadence. To test the exact positive
lineage, I continued the previous 200-step checkpoint for 400 more source
steps.

Source trajectory:

| Continuation step | Source calc | Snapshot normal | Margin loss | Source final eval |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.3225` | `0.3275` | `0.0020` | n/a |
| `200` | `0.4725` | `0.4950` | `0.0003` | n/a |
| `400` | `0.3850` | `0.4075` | `0.0000` | `0.3600` |

Geometry:

| Continuation checkpoint | Source calc | Forced best=true | True-best gap | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: |
| step `200` | `0.4725` | `0.9675` | `0.0005` | `1.1494` |
| step `400` | `0.3850` | `0.9800` | `0.0009` | `1.2038` |

Trusted handoff from the continuation step-200 checkpoint:

| Source checkpoint | Final eval | Step-600 normal | Injection-zero | Forced-random | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| continued step `200` | `0.7400` | `0.7850` | `0.0000` | `0.0300` | `0.4175` |

## Interpretation

Mixed-positive.

Longer one-negative forced-margin training improves the branch over its
200-step handoff (`0.6600`) and keeps controls low. The best final eval here is
`0.7400`; the best logged step-600 handoff snapshot is `0.7850`.

However, this does not clearly beat the scheduled forced-true step-600 source,
which reached `0.7725` final handoff. The exact prior positive lineage also
shows checkpoint sensitivity: 200 continuation steps improved source/handoff,
but 400 continuation steps degraded source final eval back to `0.3600`.

Geometry improved sharply, but was still not sufficient as a selector:
near-perfect forced-best metrics at the fresh step-600 and continued step-400
checkpoints did not guarantee the best final handoff. Actual 600-step handoff
remains the arbiter.

## Decision

```text
additive_forced_margin_long_source_mixed_positive
```

Do not repeat the same seed-13 one-negative forced-margin 600-step source
ladder, the same continuation from the prior step-200 checkpoint, or handoffs
from the tested step-400/step-600/continued-step-200 checkpoints as novelty.

Next useful test: try a late source-recovery/retention phase only if explicitly
testing whether the remaining gap is source-policy maturity. Otherwise use a
fresh seed or move toward less prescriptive/scalable assignment.
