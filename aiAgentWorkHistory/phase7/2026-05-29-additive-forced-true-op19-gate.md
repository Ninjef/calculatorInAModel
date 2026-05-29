# 2026-05-29 Additive Forced-True Schedule Op19 Gate

## Purpose

Scale the scheduled forced-true additive auxiliary from the reduced `0..9`
mechanism gate to the real `0..19` source setting, then verify with the
established standalone 600-step frozen-policy additive handoff gate.

This follows:

```text
additive_forced_true_start50_schedule_positive_small_gate
```

## Setup

Run roots:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate_steps200
```

Common source setup:

- `operand_max=19`
- `calculator_operand_vocab_size=20`
- seed `13`
- frozen product answer-decoder semantic checkpoint from 2026-05-12
- no-decay source stabilization:
  - result-policy improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`
- `answer_loss_weight=0`
- checkpoints every `100` steps

Branches:

| Branch | Aux settings |
| --- | --- |
| baseline | no additive forced-true auxiliary |
| always-on | `--additive-forced-true-loss-weight 0.5` |
| scheduled | `--additive-forced-true-loss-weight 0.5 --additive-forced-true-start-step 50` |

## Source Results

100-step source gate:

| Branch | Source train calc | Snapshot normal | Final eval exact |
| --- | ---: | ---: | ---: |
| baseline | `0.2150` | `0.2100` | `0.2200` |
| always-on | `0.2325` | `0.2225` | `0.2025` |
| scheduled | `0.2050` | `0.2050` | `0.2150` |

The 100-step gate was too early to call: scheduled improved geometry over
baseline but did not improve source accuracy.

200-step source gate:

| Branch | Source train calc | Snapshot normal | Final eval exact |
| --- | ---: | ---: | ---: |
| baseline | `0.2875` | `0.2550` | `0.2825` |
| scheduled | `0.2800` | `0.2575` | `0.2750` |

At 200 steps, source policy acquisition was nearly tied, with scheduled
slightly lower on train calc/final eval and slightly higher on snapshot normal.

## Geometry Probes

100-step checkpoint geometry:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate/geometry_probe
```

| Branch | Calc acc | Forced best=true | Forced top3=true | 50-step slope final loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.2150` | `0.0000` | `0.0000` | `1.7959` |
| always-on | `0.2325` | `0.1875` | `0.3850` | `0.8924` |
| scheduled | `0.2050` | `0.1175` | `0.3525` | `1.5045` |

200-step checkpoint geometry:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate_steps200/geometry_probe
```

| Branch | Calc acc | Forced best=true | Forced top3=true | True loss | Best loss | 50-step slope final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.2875` | `0.0000` | `0.0000` | `2.6754` | `2.6544` | `1.8058` |
| scheduled | `0.2800` | `0.2125` | `0.4025` | `1.1805` | `1.0946` | `1.0360` |

The scheduled objective preserved source policy acquisition approximately
while producing a large forced-result geometry improvement.

## Standalone 600-Step Handoff Verification

Run roots:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate_steps200/handoff600_baseline_step200
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_gate_steps200/handoff600_scheduled_step200
```

Setup:

- loaded each step-200 source checkpoint with
  `--semantic-decoder-checkpoint-load-scope compatible_model`
- `calculator_bottleneck_mode=none`
- `--freeze-calculator-policy`
- `600` downstream answer-loss steps
- full `0..19` grid

Results:

| Source | Handoff step 0 | Handoff step 600 | Final eval | Injection-zero final | Oracle final | Learned calc final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline step 200 | `0.0000` | `0.2625` | `0.2525` | `0.0000` | `0.2109` | `0.2813` |
| scheduled step 200 | `0.0425` | `0.3925` | `0.4150` | `0.0469` | `0.3672` | `0.2813` |

## Interpretation

Positive full-grid early handoff gate.

The scheduled forced-true additive source objective did not materially improve
source calculator accuracy by step 200, but it substantially improved additive
handoff behavior under the trusted 600-step verification gate. This is the
first scaled evidence that directly shaping source additive readout geometry
can improve downstream non-bottleneck transfer without merely selecting a
different checkpoint.

The absolute handoff remains weak compared with the best staged-transfer
recipes, so this is not a solved recipe. The next question is whether the
benefit compounds in longer source runs and whether a policy-retention anchor
is needed once source accuracy rises.

## Decision

```text
additive_forced_true_schedule_op19_handoff_positive
```

Do not repeat this same seed-13, `operand_max=19`, 200-step baseline vs
scheduled step-50 source gate plus 600-step handoff as novelty.

Allowed next tests:

- Extend scheduled source acquisition to a longer horizon, with checkpoints
  around `400/600/800`, then verify the best scheduled checkpoint by standalone
  600-step handoff.
- Compare start schedules only if the schedule is a real mechanism question
  such as delayed onset after source calc reaches a threshold, not just a
  minor step-50/step-75 tweak.
- Add a policy-retention anchor if longer scheduled runs improve geometry but
  drift or collapse source calculator accuracy.

