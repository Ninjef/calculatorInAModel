# 2026-05-29 Stabilized Source Continuation Boundary

## Aim

After the no-decay stabilized source passed with 600-step continuation, test
whether the continuation budget can be reduced further before the
non-bottleneck readout falls below the `0.90` gate.

The direct readout from the initial handoff had already reached only `0.8000`,
so this task probes the boundary between no continuation and the passing
600-step continuation result.

## Lineage

Base continuation run:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor/continuation_from_final_handoff_seed9_steps800
```

Readout root:

```text
runs/2026-05-29_phase7_stabilized_source_reduced_continuation
```

All readouts used 600 policy-backbone-frozen steps at LR `3e-4` from
frozen-policy continuation checkpoints.

## Result

| Continuation checkpoint | Readout final eval | Readout best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step `600` | `0.9425` | `0.9425` at readout step `300` | `0.0078` | `0.0781` | `0.9297` | `0.8750` |
| step `500` | `0.9400` | `0.9400` at readout step `300` | `0.0156` | `0.0703` | `0.9219` | `0.8750` |
| step `400` | `0.9175` | `0.9325` at readout step `500` | `0.0078` | `0.0703` | `0.9219` | `0.8750` |
| step `300` | `0.8850` | `0.9150` at readout step `500` | `0.0078` | `0.0625` | `0.9063` | `0.8750` |

For context:

| Reference | Final eval |
| --- | ---: |
| direct readout from initial handoff | `0.8000` |
| 800 continuation + 600 readout | `0.9575` |

## Decision

Label:

```text
stabilized_source_400_continuation_boundary_positive_300_negative
```

For this no-decay stabilized lineage, 400 frozen-policy continuation steps are
enough to clear the non-bottleneck gate after 600-step readout, but 300 steps
are not enough by final eval.

## Interpretation

This narrows the continuation budget for the current best source family:
800 is no longer necessary on this lineage, 600 and 500 pass comfortably, and
400 still passes with reduced margin. The failed 300-step final eval says not
to collapse continuation into a token gesture; the readout still needs a few
hundred continuation steps to organize the additive answer path around the
frozen calculator signal.

The step-300 readout did briefly hit `0.9150`, but final eval fell to `0.8850`.
Until there is a validated model-selection rule for intermediate readout
snapshots, the gate should use final eval or a predeclared selector.

## Anti-Rerun Note

Do not repeat this exact no-decay stabilized continuation ladder
(`600/500/400/300` checkpoints into 600-step readout) as novelty.

Next useful tests:

- replicate the 400/500 continuation boundary on another no-decay stabilized
  source;
- test a predeclared readout-snapshot selector if using best snapshots rather
  than final eval;
- build a cheaper proxy for choosing continuation budget before running the
  full readout.

## Verification

The step `500`, `400`, and `300` continuation readouts completed and wrote
metrics under the reduced-continuation run root. The step `600` readout was
recorded in the previous reduced-continuation task.
