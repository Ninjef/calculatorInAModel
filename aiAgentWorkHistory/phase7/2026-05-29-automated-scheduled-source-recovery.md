# 2026-05-29 Automated Scheduled Source Recovery

## Purpose

The previous two experiments showed that a manual late-source recovery phase
can restore learned calculator quality after scheduled forced-true geometry
formation. This experiment removes the manual checkpoint relaunch by adding an
in-run recovery switch to source training.

Question:

Can a single source run switch at step `600` from high-pressure geometry
formation to low-LR/lower-aux recovery and preserve the trusted handoff result?

## Tooling

Added non-default flags to `scripts/overfit_one_batch.py`:

```text
--late-source-recovery-start-step
--late-source-recovery-lr-multiplier
--late-source-recovery-additive-forced-true-loss-weight
```

Default behavior is unchanged. When enabled, the script lowers optimizer group
LRs by the multiplier and optionally overrides the additive forced-true weight
from the configured step onward.

## Automated Source Run

Run root:

```text
runs/2026-05-29_phase7_scheduled_source_auto_recovery/seed14_auto_late_recovery_steps630_cpu
```

Configuration:

- CPU device
- CLI seed `14` (saved seed `16`)
- `operand_max=19`, exhaustive grid
- bottleneck source mode with frozen product answer decoder
- hard improvement assignment weight `10`
- entropy `0.05`, batch diversity `0.1`
- scheduled forced-true additive auxiliary:
  - weight `0.5`
  - start step `50`
- late-source recovery:
  - start step `600`
  - LR multiplier `0.1`
  - forced-true additive weight override `0.1`
- `630` source steps

Source trajectory:

| Step | Source normal / calc | Injection-zero | Oracle | Forced-true effective weight | Recovery LR multiplier |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0275` | `0.0550` | `1.0000` | n/a | `1.0` |
| `100` | `0.2425` | `0.0600` | `1.0000` | `0.5` | `1.0` |
| `200` | `0.3775` | `0.0450` | `1.0000` | `0.5` | `1.0` |
| `300` | `0.5300` | `0.0625` | `1.0000` | `0.5` | `1.0` |
| `400` | `0.5600` | `0.0575` | `1.0000` | `0.5` | `1.0` |
| `500` | `0.6275` | `0.0600` | `1.0000` | `0.5` | `1.0` |
| `600` | `0.6375` | `0.0575` | `1.0000` | `0.1` | `0.1` |

Final source eval was `0.8775`, compared with `0.8850` for the previous manual
30-step recovery relaunch.

## Trusted Handoff Gate

Run root:

```text
runs/2026-05-29_phase7_scheduled_source_auto_recovery/seed14_handoff600_from_auto_late_recovery_cpu
```

Configuration:

- loaded automated source final checkpoint with `compatible_model` scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` downstream steps

Handoff trajectory:

| Step | Normal | Injection-zero | Forced-random | Oracle | Learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.7675` | `0.0550` | `0.0550` | `0.8600` | `0.8700` |
| `100` | `0.8375` | `0.0675` | `0.0650` | `0.8525` | `0.8675` |
| `200` | `0.8800` | `0.1025` | `0.1000` | `0.9350` | `0.8725` |
| `300` | `0.9125` | `0.0825` | `0.0850` | `0.9475` | `0.8850` |
| `400` | `0.9175` | `0.1000` | `0.0525` | `0.9525` | `0.8775` |
| `500` | `0.9275` | `0.1150` | `0.0600` | `0.9750` | `0.8750` |
| `600` | `0.9475` | `0.0800` | `0.0775` | `0.9650` | `0.8725` |

Final eval was `0.9400`.

## Decision

```text
automated_scheduled_source_recovery_positive
```

## Interpretation

Positive.

The automated in-run phase switch preserved the late-source recovery effect
without manual checkpoint selection/relaunch. The resulting handoff is a little
below the manual seed-14 relaunch (`0.9400` vs `0.9600` final eval), but it
still clears the high non-bottleneck gate with low zero/random controls.

This improves the staged recipe ergonomically and conceptually: the recovery
can be part of source training. It does not solve the full project goal because
the method still uses hard improvement assignment, a true-result forced
auxiliary, and a fixed hand-tuned transition step.

## Anti-Rerun Note

Do not repeat this exact seed-14 fixed-step-600 automated recovery plus
600-step handoff as novelty.

Next useful work:

- use an adaptive transition criterion instead of fixed step `600`;
- test whether late recovery can be selected from source/geometry metrics
  without trusted handoff leakage;
- move back toward less prescriptive or lower-cost assignment while preserving
  the trusted handoff/readout gates.
