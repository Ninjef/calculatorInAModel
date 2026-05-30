# 2026-05-30 op29 Low-LR Source Recovery Diagnostic

## Question

Was the op29 range-stress miss mainly caused by an under-mature source policy,
or does the larger range require a different mechanism?

This is a diagnostic continuation, not a new forced-margin recipe. It continues
the failed op29 product source checkpoint from step `630` with the same style
of low-LR reduced-margin recovery that previously rescued the op19
one-negative forced-margin source.

## Runs

Source recovery:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_low_lr_recovery_from_step630_steps90_cpu
```

Configuration:

- loaded op29 product forced-margin source step `630` with `compatible_model`
- `operand_max=29`, exhaustive `30 x 30` grid
- LR `0.0003`
- one-negative forced-margin retained at weight `0.1`
- source stabilization retained:
  - result-policy entropy `0.05`
  - batch diversity `0.1`
  - improvement assignment weight `10`
- `90` CPU recovery steps

Trusted handoff:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_handoff600_from_low_lr_recovery_step90_cpu
```

Configuration:

- loaded recovered step `90` checkpoint with `compatible_model`
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` CPU steps

## Results

Source recovery:

| Recovery step | Source calc | Snapshot normal | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.6789` | `0.6789` | `0.0278` | `1.0000` | `0.0122` |
| `30` | `0.7144` | `0.7144` | `0.0256` | `1.0000` | `0.0122` |
| `60` | `0.8089` | `0.8089` | `0.0289` | `1.0000` | `0.0100` |
| `90` | `0.8211` | `0.8211` | `0.0256` | `1.0000` | `0.0178` |

Final source eval was `741/900 = 0.8233`.

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.8522` | `0.0322` | `0.9767` | `0.8611` | `0.0200` |
| `200` | `0.8533` | `0.0378` | `0.9811` | `0.8378` | `0.0089` |
| `300` | `0.8700` | `0.0256` | `0.9778` | `0.8433` | `0.0178` |
| `400` | `0.8867` | `0.0256` | `0.9578` | `0.8278` | `0.0144` |
| `500` | `0.9189` | `0.0256` | `0.9644` | `0.8489` | `0.0089` |
| `600` | `0.8978` | `0.0122` | `0.9611` | `0.8233` | `0.0111` |

Final handoff eval was `816/900 = 0.9067`. The 128-sample diagnostic summary
reported normal `0.9219`, injection-zero `0.0078`, forced-random `0.0000`,
and learned calculator accuracy `0.8438`.

## Decision

```text
op29_low_lr_source_recovery_diagnostic_mixed_positive
```

Interpretation:

- The op29 range miss was partly source-policy-maturity limited. A gentle
  recovery phase lifted source calc from `0.6889` to `0.8211` and improved
  trusted handoff from `0.8533` final / `0.8278` step-600 normal to `0.9067`
  final / `0.8978` step-600 normal.
- This does not make the current method scalable. The rescue adds another
  prescriptive full-grid source continuation on top of the already expensive
  hard-assignment forced-margin recipe.
- The handoff remains calculator-causal: injection-zero and forced-random
  controls stay very low while normal accuracy is high.

Do not rerun this same op29 step-630 to low-LR step-90 recovery and handoff as
novelty. Further range work should either change source acquisition, reduce
assignment cost against an exact-grid ceiling, or test a materially different
capacity/recovery hypothesis rather than extending this same continuation
ladder.
