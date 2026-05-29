# 2026-05-29 Additive Forced-True Schedule Gate

## Purpose

Follow up the always-on forced-true additive auxiliary result:

```text
additive_forced_true_source_aux_mixed_positive_small_gate
```

The always-on auxiliary shaped additive readout geometry, but weakened source
policy acquisition. This gate tests a new mechanism, not a weight tweak: delay
the geometry objective until the source policy has begun learning.

## Implementation

Extended `scripts/overfit_one_batch.py` with:

```text
--additive-forced-true-start-step
--additive-forced-true-ramp-steps
```

Effective weight:

```text
0 before start_step
full weight after start_step when ramp_steps=0
linear ramp from 0 to full weight when ramp_steps>0
```

Smoke:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/smoke
```

The smoke used `start_step=1`, `ramp_steps=1`, and confirmed that the auxiliary
stayed off at steps `0/1` and turned on at step `2`.

## Gate

Run root:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/small_gate
```

Shared setup with the previous small gate:

- `operand_max=9`
- `calculator_operand_vocab_size=20`
- seed `13`
- `100` source steps
- frozen semantic decoder from the 2026-05-12 product answer-decoder checkpoint
- no-decay source stabilization:
  - result-policy improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`

New branch:

```text
--additive-forced-true-loss-weight 0.5
--additive-forced-true-start-step 50
```

## Results

Source-training comparison:

| Branch | Aux schedule | Source calc @100 | Source normal snapshot | Final eval exact |
| --- | --- | ---: | ---: | ---: |
| baseline | none | `0.3500` | `0.3400` | `0.3800` |
| always-on aux | weight `0.5` from step `0` | `0.2800` | `0.3200` | `0.2800` |
| scheduled aux | weight `0.5` from step `50` | `0.3900` | `0.4300` | `0.4000` |

Geometry probe on step-100 checkpoints:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/small_gate/geometry_probe
```

| Branch | Calc acc | Forced best=true | Forced top3=true | True loss | Best loss | 50-step slope final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.3500` | `0.0000` | `0.0000` | `2.6463` | `2.6265` | `1.5305` |
| always-on aux | `0.2800` | `0.5900` | `0.6900` | `0.8043` | `0.7881` | `0.7367` |
| scheduled aux | `0.3900` | `0.5100` | `0.5600` | `1.3888` | `1.3633` | `0.7979` |

## Interpretation

Positive small-gate mechanism.

Delaying the forced-true additive auxiliary to step `50` avoided the source
policy acquisition penalty observed with the always-on version and preserved a
large additive-geometry improvement over baseline. The scheduled branch beat
the baseline on source calculator accuracy, source normal snapshot, final eval,
forced-result geometry, and 50-step downstream slope.

This is still a reduced `0..9` grid and still prescriptive. It does not prove
the final thesis. It does justify a larger `operand_max=19` gate with targeted
standalone 600-step additive handoff verification.

## Decision

```text
additive_forced_true_start50_schedule_positive_small_gate
```

Do not repeat this same `operand_max=9`, seed-13, 100-step, start-step-50
schedule gate as novelty.

Allowed next tests:

- Run an `operand_max=19` source-only checkpoint gate with scheduled
  forced-true auxiliary and verify promising checkpoints with standalone
  600-step additive handoff.
- Test a behavior-gated version where the auxiliary turns on once source
  calculator accuracy or anchor agreement crosses a predeclared threshold.
- Pair the scheduled geometry objective with a policy-retention anchor if a
  larger run shows late source-policy drift.

