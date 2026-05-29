# 2026-05-29 Fresh Scheduled Source Recovery Replication

## Purpose

Replicate the seed-13 scheduled-source low-LR recovery result on a fresh seed.
The prior run showed that a 30-step late-source phase with LR `3e-4` and
forced-true additive weight `0.1` restored learned calculator accuracy and
eventually cleared readout. This run tests whether that was a one-seed rescue
or a reusable scheduled-source phase.

## Fresh Scheduled Source

Initial MPS launch wrote only a config and did not produce curves/checkpoints
in a useful time window, matching the local MPS stall seen in the previous
turn. I stopped it and reran on CPU with `--device cpu`.

CPU run root:

```text
runs/2026-05-29_phase7_scheduled_source_fresh_recovery/seed14_scheduled_steps600_cpu
```

Configuration:

- CLI seed `14` (saved run directory seed `16`)
- `operand_max=19`, exhaustive grid
- bottleneck source mode with frozen product answer decoder
- hard improvement assignment weight `10`
- entropy `0.05`, batch diversity `0.1`
- scheduled forced-true additive auxiliary:
  - weight `0.5`
  - start step `50`
- `600` source steps

Source trajectory:

| Step | Source normal / calc | Injection-zero | Oracle | Forced-true additive loss |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0275` | `0.0550` | `1.0000` | n/a |
| `100` | `0.2425` | `0.0600` | `1.0000` | `1.5532` |
| `200` | `0.3775` | `0.0450` | `1.0000` | `0.7745` |
| `300` | `0.5300` | `0.0625` | `1.0000` | `0.5420` |
| `400` | `0.5600` | `0.0575` | `1.0000` | `0.2852` |
| `500` | `0.6275` | `0.0600` | `1.0000` | `0.2309` |
| `600` | `0.6375` | `0.0575` | `1.0000` | `0.2202` |

Final source eval was `0.6675`.

## Low-LR Recovery

Run root:

```text
runs/2026-05-29_phase7_scheduled_source_fresh_recovery/seed14_low_aux0p1_recovery_from_step600_steps30_cpu
```

Configuration:

- loaded scheduled step-600 source checkpoint with `full_model` scope
- same source stabilization objective
- LR `0.0003`
- forced-true additive weight reduced to `0.1`
- `30` steps

Recovery trajectory:

| Step | Source normal / calc | Injection-zero | Oracle | Forced-true additive loss |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.6700` | `0.0550` | `1.0000` | `0.2202` |
| `10` | `0.8000` | `0.0450` | `1.0000` | `0.1504` |
| `20` | `0.8825` | `0.0375` | `1.0000` | `0.1480` |
| `30` | `0.8900` | `0.0425` | `1.0000` | `0.1433` |

Final source eval was `0.8850`.

## Trusted Handoff Gate

Run root:

```text
runs/2026-05-29_phase7_scheduled_source_fresh_recovery/seed14_handoff600_from_recovery_step30_cpu
```

Configuration:

- loaded recovered step-30 source checkpoint with `compatible_model` scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` downstream steps

Handoff trajectory:

| Step | Normal | Injection-zero | Forced-random | Oracle | Learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.7750` | `0.0650` | `0.0625` | `0.8900` | `0.8575` |
| `100` | `0.8725` | `0.0675` | `0.0575` | `0.9250` | `0.8850` |
| `200` | `0.8825` | `0.0800` | `0.0925` | `0.9600` | `0.8575` |
| `300` | `0.9050` | `0.0900` | `0.0600` | `0.9650` | `0.8675` |
| `400` | `0.9150` | `0.1000` | `0.0675` | `0.9625` | `0.8475` |
| `500` | `0.9400` | `0.0975` | `0.0575` | `0.9600` | `0.8900` |
| `600` | `0.9650` | `0.0850` | `0.0875` | `0.9850` | `0.8700` |

Final eval was `0.9600`.

## Decision

```text
fresh_scheduled_source_recovery_replication_positive
```

## Interpretation

Positive.

The gentle scheduled-source recovery recipe replicated on a fresh seed and
cleared the high non-bottleneck handoff gate directly. The recovery phase
improved both source calculator accuracy and forced-true additive loss, then
the frozen-policy additive downstream path reached `0.9600` final eval.

The seed-14 zero/random controls are higher than the seed-13 readout controls,
but they remain far below normal and oracle, so the result is still strongly
calculator-dependent.

This strengthens the next direction: automate the late-source transition
instead of manually choosing step `600` and relaunching with lower LR and
lower auxiliary weight.

## Anti-Rerun Note

Do not repeat this exact seed-14 chain as novelty:

```text
scheduled source to step 600 -> 30-step low-LR aux=0.1 recovery -> 600-step
frozen-policy handoff
```

Next useful work:

- implement or script an automatic late-source recovery phase;
- test a third seed only if the question is explicit stability;
- preserve the 600-step handoff gate and monitor zero/random controls, since
  seed 14 had higher bypass controls than seed 13.
