# 2026-05-29 Seed-10 Source Checkpoint Geometry Sweep

## Aim

Test whether the seed-10 no-decay stabilized transfer failure was mainly a
bad final-checkpoint selection, or whether the lineage has broadly weak
non-bottleneck transfer geometry.

This follows the anti-rerun note from the seed-10 replication task: do not
repeat the final-source handoff, but compare seed-9 positive and seed-10
negative geometry or build a cheaper transfer proxy.

## Runs

Run root:

```text
runs/2026-05-29_phase7_seed10_source_checkpoint_handoff_sweep
```

Seed-10 checkpoint handoff probes:

| Source checkpoint | Handoff run |
| --- | --- |
| step `1000` | `step1000_handoff600` |
| step `1300` | `step1300_handoff600` |
| step `1400` | `step1400_handoff600` |

Reference runs:

- positive seed-9 final-source handoff:
  `runs/2026-05-29_phase7_source_acquisition_stabilization_floor/handoff_final_additive_seed9`
- negative seed-10 final-source handoff:
  `runs/2026-05-29_phase7_stabilized_source_replication/handoff_final_additive_seed10`

Cheap proxy probe:

```text
runs/2026-05-29_phase7_seed10_source_checkpoint_handoff_sweep/frozen_state_probe
```

The proxy used the existing frozen-state linear probe over `read_pair` and
`layer2_pair`.

## Handoff Result

| Source | 600-step handoff snapshot | Final eval | Injection-zero | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed-9 final reference | `0.5250` | `0.6500` | `0.0150` | `0.5275` | `0.8750` |
| seed-10 step `1000` | `0.4475` | `0.4450` | `0.0125` | `0.4175` | `0.7300` |
| seed-10 step `1300` | `0.4325` | `0.4200` | `0.0725` | `0.4250` | `0.8275` |
| seed-10 step `1400` | `0.4225` | `0.4025` | `0.0175` | `0.4250` | `0.8625` |
| seed-10 final reference | `0.3375` | `0.3275` | `0.0650` | `0.3300` | `0.9250` |

Earlier seed-10 checkpoints transfer better than the final checkpoint, so
source-checkpoint selection still matters. But none of them approach the
seed-9 positive lineage, and handoff quality degrades while learned calculator
accuracy improves.

## Frozen-State Proxy Result

| Source | Best frozen-state feature | Best eval accuracy | Mean eval accuracy |
| --- | --- | ---: | ---: |
| seed-9 final | `read_pair` | `0.3625` | `0.3563` |
| seed-10 step `1000` | `read_pair` | `0.2750` | `0.2750` |
| seed-10 step `1300` | `read_pair` | `0.3125` | `0.3063` |
| seed-10 step `1400` | `layer2_pair` | `0.4375` | `0.4250` |
| seed-10 final | `layer2_pair` | `0.4500` | `0.4250` |

The frozen linear sum probe is not a valid transfer proxy here: it ranks the
seed-10 final checkpoint highest even though that checkpoint has the worst
handoff. Linear recoverability of the sum from source states is not the same
as additive answer-path usability.

## Decision

Label:

```text
seed10_checkpoint_selection_partial_geometry_negative
```

The seed-10 failure is not just a final-checkpoint artifact. Earlier
checkpoints improve handoff, with step `1000` best in this sweep, but the
lineage remains transfer-weak compared with seed-9. The cheap frozen-state
linear readout probe is also insufficient as a selector.

## Interpretation

The most useful next proxy should measure additive learning slope or the
local injection-to-answer geometry directly. It should not measure only:

- bottleneck source accuracy;
- learned calculator accuracy;
- oracle-at-eval recovery;
- linear sum decodability from frozen source states.

The striking pattern is that seed-10 calculator accuracy rises from `0.7300`
at step `1000` to `0.9250` at final, while handoff falls from `0.4475` to
`0.3375`. The source objective can keep improving the hard calculator policy
while making the additive readout geometry worse.

## Anti-Rerun Note

Do not repeat the seed-10 step `1000`/`1300`/`1400` 600-step handoff sweep or
the frozen-state linear probe over these same checkpoints as novelty.

Next useful tests:

- build an injection-to-answer geometry metric at handoff start;
- run a tiny additive learning-slope probe and validate it against existing
  seed-9/seed-10 outcomes;
- optimize source acquisition for early handoff slope rather than only source
  calculator accuracy.

## Verification

All three new handoff probes completed and wrote metrics under the run root.
The frozen-state probe also completed and wrote
`frozen_state_probe_summary_all.json`.
