# Phase 7 Eightieth Task: Seed-10 Source Checkpoint Geometry Sweep

## Status

Completed 2026-05-29.

## Question

Was the seed-10 no-decay stabilized transfer failure mainly caused by choosing
the final source checkpoint, and can the existing frozen-state linear readout
probe serve as a cheap transfer proxy?

## Setup

- Ran 600-step frozen-policy additive handoff probes from seed-10 source
  checkpoints `1000`, `1300`, and `1400`.
- Compared them with the existing seed-10 final-source handoff and the
  seed-9 positive final-source handoff.
- Ran the existing frozen-state linear readout probe on seed-9 final plus
  seed-10 `1000`/`1300`/`1400`/final checkpoints.

## Result

| Source | 600-step handoff | Final eval | Calc |
| --- | ---: | ---: | ---: |
| seed-9 final reference | `0.5250` | `0.6500` | `0.8750` |
| seed-10 step `1000` | `0.4475` | `0.4450` | `0.7300` |
| seed-10 step `1300` | `0.4325` | `0.4200` | `0.8275` |
| seed-10 step `1400` | `0.4225` | `0.4025` | `0.8625` |
| seed-10 final reference | `0.3375` | `0.3275` | `0.9250` |

Frozen-state linear probe best eval accuracy:

| Source | Best eval |
| --- | ---: |
| seed-9 final | `0.3625` |
| seed-10 step `1000` | `0.2750` |
| seed-10 step `1300` | `0.3125` |
| seed-10 step `1400` | `0.4375` |
| seed-10 final | `0.4500` |

## Decision

```text
seed10_checkpoint_selection_partial_geometry_negative
```

Earlier seed-10 checkpoints are better than final for additive handoff, but
none approach the seed-9 positive lineage. The frozen-state linear probe is
not a valid selector because it ranks seed-10 final highest despite the worst
handoff.

## Next

Build or test a proxy that measures additive learning slope or
injection-to-answer geometry directly.
