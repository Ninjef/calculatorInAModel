# 2026-05-30 Automated Forced-Margin Source Recovery

## Question

Can the one-negative forced-margin low-LR recovery effect be folded into one
source run and replicate on a fresh seed, instead of relying on a manual
checkpoint continuation?

This follows the prior manual recovery gate without rerunning it: the mechanism
change is an in-run late recovery override for the additive forced-margin
weight, and the empirical check uses a fresh source seed.

## Tooling

Added:

```text
--late-source-recovery-additive-forced-margin-loss-weight
```

When late source recovery is active, this overrides the additive forced-margin
weight just as the existing forced-true override does. The run metadata records
both the configured override and the final effective forced-margin weight.

Focused verification:

```text
PYTHONPATH=. pytest tests/test_model.py -q -k 'late_source_recovery or additive_forced_margin'
```

Result: `2 passed, 117 deselected`.

## Runs

Fresh-seed automated source:

```text
runs/2026-05-30_phase7_forced_margin_auto_recovery/fresh_seed16_source630_cpu
```

Configuration:

- full-grid source mode, `operand_max=19`
- one-negative forced-margin, initial weight `0.5`, start step `50`
- source stabilization retained:
  - result-policy entropy `0.05`
  - batch diversity `0.1`
  - improvement assignment weight `10`
- late recovery starts at step `600`
  - LR multiplier `0.1`
  - forced-margin weight override `0.1`
- `630` CPU source steps
- seed `16`

Trusted handoff:

```text
runs/2026-05-30_phase7_forced_margin_auto_recovery/handoff600_from_fresh_seed16_auto_step630_cpu
```

Configuration:

- loaded source step `630` with `compatible_model` scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` CPU steps

## Results

Source:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Late recovery |
| ---: | ---: | ---: | ---: | ---: | --- |
| `570` | `0.5800` | `0.6125` | `0.0600` | `1.0000` | off |
| `600` | `0.5825` | `0.5725` | `0.0425` | `1.0000` | on |
| `630` | `0.8825` | `0.9000` | `0.0425` | `1.0000` | on |

Final source eval was `0.8700`. The run metadata confirms
`final_late_source_recovery_active=true`, final LR multiplier `0.1`, and final
forced-margin effective weight `0.1`. Diagnostic source learned calc was
`0.8906`, injection-zero `0.0547`, forced-random `0.0391`.

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle |
| ---: | ---: | ---: | ---: |
| `100` | `0.9125` | `0.0075` | `0.9850` |
| `200` | `0.9300` | `0.0550` | `0.9775` |
| `300` | `0.9600` | `0.0275` | `0.9900` |
| `400` | `0.9850` | `0.0175` | `0.9625` |
| `500` | `0.9650` | `0.0100` | `0.9625` |
| `600` | `0.9800` | `0.0250` | `0.9625` |

Final eval was `0.9875`. Diagnostic learned calc was `0.8906`,
injection-zero `0.0156`, forced-random `0.0938`, and oracle-at-eval `0.9844`.

## Decision

```text
automated_forced_margin_recovery_fresh_seed_positive
```

Interpretation:

- The manual forced-margin recovery effect survives automation and a fresh
  source seed.
- The late phase sharply improved source calculator accuracy in the intended
  window (`0.5825 -> 0.8825`) and produced the strongest forced-margin handoff
  so far (`0.9875` final).
- This is a strong staged-transfer result, but it is still not the final goal:
  source training remains prescriptive through hard improvement assignment and
  true-result forced-margin pressure.
- Do not rerun the same seed-16 automated recovery plus handoff as novelty.
  Future forced-margin work should either test broader stability/scale or use
  this result as evidence while moving toward less-prescriptive credit
  assignment.
