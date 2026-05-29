# 2026-05-29 Additive Forced-True Long Source Gate

## Purpose

Extend the scheduled forced-true additive source objective to longer full-grid
source horizons after the 200-step `operand_max=19` gate improved standalone
handoff from `0.2525` to `0.4150`.

Question:

Does scheduled source acquisition continue improving additive handoff geometry
through `400/600/800` source steps, or does it drift like prior high-accuracy
source branches?

## Source Run

Run root:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long
```

Setup:

- `operand_max=19`
- seed `13`
- frozen product answer-decoder semantic checkpoint from 2026-05-12
- no-decay source stabilization:
  - result-policy improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`
- `--additive-forced-true-loss-weight 0.5`
- `--additive-forced-true-start-step 50`
- source checkpoints every `200` steps

Source trajectory:

| Step | Source calc / snapshot normal | Injection-zero | Oracle | Forced-true additive loss |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0200` | `0.0675` | `1.0000` | n/a |
| `200` | `0.2575` | `0.0425` | `1.0000` | `1.1805` |
| `400` | `0.2900` | `0.0525` | `1.0000` | `0.6094` |
| `600` | `0.5825` | `0.0375` | `1.0000` | `0.0268` |
| `800` | `0.5800` | `0.0525` | `1.0000` | `0.0009` |

Final source eval exact-match was `0.5175`.

## Geometry Probe

Probe root:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/geometry_probe
```

| Source step | Calc acc | Forced best=true | Forced top3=true | True loss | Best loss | 50-step slope final loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `200` | `0.2800` | `0.2125` | `0.4025` | `1.1805` | `1.0946` | `1.0360` |
| `400` | `0.2825` | `0.4925` | `0.8225` | `0.6094` | `0.5916` | `0.7940` |
| `600` | `0.5925` | `0.9800` | `0.9975` | `0.0268` | `0.0253` | `0.4719` |
| `800` | `0.5525` | `1.0000` | `1.0000` | `0.0009` | `0.0009` | `0.5535` |

Interpretation before handoff:

- Geometry improved monotonically through step `800` by forced-result metrics.
- The 50-step downstream slope and source calc both favored step `600` over
  step `800`.
- Therefore I verified both step `600` and step `800` with standalone handoff.

## Standalone 600-Step Handoff Verification

Run roots:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/handoff600_step600
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/handoff600_step800
```

Setup:

- loaded each source checkpoint with compatible model scope
- `calculator_bottleneck_mode=none`
- `--freeze-calculator-policy`
- `600` downstream answer-loss steps

Results:

| Source checkpoint | Handoff step 0 | Best logged handoff | Handoff step 600 | Final eval | Injection-zero final | Oracle final | Learned calc final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `600` | `0.5775` | `0.7150` @ step `500` | `0.6975` | `0.7725` | `0.0469` | `0.7344` | `0.5391` |
| step `800` | `0.5300` | `0.7225` @ step `400` | `0.6700` | `0.6750` | `0.0156` | `0.5938` | `0.4219` |

For comparison, the prior 200-step scheduled source checkpoint reached only
`0.4150` final handoff, while the matched 200-step baseline reached `0.2525`.

## Decision

```text
additive_forced_true_schedule_long_source_positive_step600_best
```

## Interpretation

Positive.

The scheduled forced-true additive source objective continues improving
transfer-relevant geometry well beyond the 200-step gate. The step-600 source
checkpoint produces a much stronger standalone frozen-policy additive handoff
(`0.7725` final eval) than the 200-step scheduled checkpoint (`0.4150`) and
the matched 200-step baseline (`0.2525`).

However, final source checkpoint selection still matters. Step `800` had
perfect forced-result geometry but worse handoff than step `600`. This means
forced-result geometry alone should not replace the standalone handoff gate;
use it as a triage/diagnostic signal, then verify promising checkpoints with
the trusted 600-step handoff.

## Anti-Rerun Note

Do not repeat this same seed-13 scheduled source `200/400/600/800` geometry
ladder or the step-600/step-800 handoff comparison as novelty.

Allowed next tests:

- Run continuation/readout from the step-600 handoff lineage to see whether the
  scheduled source can clear the high non-bottleneck gate.
- Replicate scheduled source acquisition on a fresh seed only if the explicit
  question is seed stability.
- Add a policy-retention or behavior gate only if longer source horizons show
  source accuracy drift or if continuation/readout exposes calculator-policy
  degradation.

