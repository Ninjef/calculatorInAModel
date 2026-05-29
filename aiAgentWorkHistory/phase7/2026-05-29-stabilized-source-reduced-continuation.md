# 2026-05-29 Stabilized Source Reduced Continuation

## Aim

Reduce the continuation cost for the no-decay stabilized source lineage that
cleared the non-bottleneck gate with 800 frozen-policy continuation plus
600-step readout.

The earlier reduced-continuation result was source-sensitive on older
`src4/src5` lineages, so this is a new test for the no-decay stabilized source
family rather than a rerun.

## Lineage

Base continuation run:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor/continuation_from_final_handoff_seed9_steps800
```

Reduced-continuation checkpoint:

```text
checkpoint_snapshots/step_00600_weights.pt
```

Readout run:

```text
runs/2026-05-29_phase7_stabilized_source_reduced_continuation/readout_from_continuation_step600_seed9
```

## Result

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 600-step continuation snapshot | n/a | `0.8850` at step `600` | `0.0000` | n/a | `0.8800` | `0.8750` |
| 600-step readout from step `600` | `0.9425` | `0.9425` at step `300` | `0.0078` | `0.0781` | `0.9297` | `0.8750` |

For comparison, the full 800-step continuation plus 600-step readout reached
`0.9575`. The reduced continuation loses `0.0150` final eval but still clears
the `0.90` non-bottleneck gate comfortably.

## Decision

Label:

```text
stabilized_source_600_continuation_readout_positive
```

The no-decay stabilized source does not require the full 800-step continuation
to pass. A 600-step frozen-policy continuation plus 600-step readout is enough
for this lineage.

## Interpretation

The result improves the scalability profile of the current non-bottleneck
recipe for stabilized sources. It also refines the earlier
source-sensitive reduced-continuation finding: weak selected `src4` failed with
600 continuation, but this no-decay stabilized lineage passes with a large
margin.

This should not be generalized to all sources yet. It needs fresh-source
replication before replacing the 800-step continuation default.

## Anti-Rerun Note

Do not repeat this exact readout from the no-decay stabilized continuation
step-600 checkpoint as novelty.

Next useful tests:

- replicate no-decay stabilized source acquisition plus 600-continuation
  readout on another fresh seed;
- test whether 500-step continuation is enough for this stabilized lineage;
- build a cheap proxy for deciding whether a source can use 600 continuation
  or needs the full 800.

## Verification

The 600-step readout completed and wrote metrics under the reduced-continuation
run root above.
