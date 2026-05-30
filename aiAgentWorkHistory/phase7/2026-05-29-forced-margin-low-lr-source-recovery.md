# 2026-05-29 Forced-Margin Low-LR Source Recovery

## Question

Is the one-negative forced-margin branch mainly limited by source-policy
maturity, the same way the scheduled forced-true branch was before low-LR
source recovery?

This was a predeclared allowed forced-margin follow-up after the branch review:
do not rerun longer same-seed ladders, but do test a source recovery/retention
phase if it directly checks whether the checkpoint is under-mature.

## Runs

Source recovery:

```text
runs/2026-05-29_phase7_forced_margin_source_recovery/lr3e4_margin0p1_from_long_step600_steps30_cpu
```

Configuration:

- loaded the longer one-negative forced-margin source step-600 checkpoint
- bottleneck source mode, full exhaustive grid
- LR `0.0003`
- one-negative forced-margin retained but reduced from weight `0.5` to `0.1`
- existing source stabilization retained:
  - result-policy entropy `0.05`
  - batch diversity `0.1`
  - improvement assignment weight `10`
- `30` CPU steps

Trusted handoff:

```text
runs/2026-05-29_phase7_forced_margin_source_recovery/handoff600_from_lr3e4_margin0p1_step30_cpu
```

Configuration:

- loaded recovered source step-30 checkpoint with `compatible_model` scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` CPU steps

## Results

Source recovery from the unrecovered forced-margin step-600 checkpoint:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.5225` | `0.5075` | `0.0500` | `1.0000` | `0.0078` |
| `10` | `0.7825` | `0.7750` | `0.0500` | `1.0000` | n/a |
| `20` | `0.7350` | `0.7175` | `0.0600` | `1.0000` | n/a |
| `30` | `0.7725` | `0.7400` | `0.0400` | `1.0000` | n/a |

Final source eval was `0.7825`; the diagnostic learned calculator-result
accuracy was `0.8594`, with injection-zero `0.0703` and forced-random
`0.0078`.

Trusted 600-step frozen-policy handoff from recovered step 30:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0000` | `0.0000` | `0.0000` | n/a |
| `100` | `0.6525` | `0.0350` | `0.8375` | n/a |
| `200` | `0.7850` | `0.0225` | `0.9300` | n/a |
| `300` | `0.8225` | `0.0050` | `0.9375` | n/a |
| `400` | `0.8525` | `0.0050` | `0.9375` | n/a |
| `500` | `0.8825` | `0.0000` | `0.9475` | n/a |
| `600` | `0.9050` | `0.0000` | `0.9375` | `0.8594` |

Final handoff eval was `0.8700`; diagnostic exact match was `0.8828`,
calculator-result accuracy `0.8594`, injection-zero `0.0000`, forced-random
`0.0313`, and oracle-at-eval `0.9375`.

## Comparison

- Unrecovered longer one-negative forced-margin step-600 handoff:
  `0.7330` final / `0.7500` step-600 normal.
- Best prior longer forced-margin handoff:
  `0.7400` final / `0.7850` step-600 normal.
- Scheduled forced-true step-600 handoff before low-LR recovery:
  `0.7725` final.
- Forced-margin low-LR recovery handoff:
  `0.8700` final / `0.9050` step-600 normal.
- Automated scheduled-source recovery remains stronger:
  `0.9400` final / `0.9475` step-600 normal.

## Decision

```text
forced_margin_low_lr_recovery_positive_but_prescriptive
```

Interpretation:

- The one-negative forced-margin branch was partly source-policy-maturity
  limited. A gentle late low-LR phase dramatically improved both source
  calculator accuracy and trusted non-bottleneck handoff.
- This rescues forced-margin as a useful source-acquisition auxiliary, not as a
  final scalable solution. It still uses hard assignment and true-result
  contrastive forcing, so it remains prescriptive.
- Do not rerun this exact `lr3e-4`, margin-weight `0.1`, step-600-to-step-30
  seed-15 recovery plus 600-step handoff as novelty.
- Next useful forced-margin work is either fresh-seed stability for this
  recovery pattern or folding the recovery into an automated source run.
  Otherwise prioritize less-prescriptive target construction or estimator work.
