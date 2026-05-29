# 2026-05-29 Source Acquisition Stabilization Variant

## Aim

Test whether small entropy plus batch-diversity regularization can stabilize a
fresh bottleneck source policy enough to improve the weak-source boundary seen
in `src7`, before spending downstream non-bottleneck handoff compute.

## Source Acquisition

Run root:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_variant
```

Source cell:

```text
src7_entropy0p05_div0p1_decay1200_steps1600
```

Saved run:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_variant/src7_entropy0p05_div0p1_decay1200_steps1600/2026-05-29_005335_835739_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed9
```

Configuration matched the current bottleneck source-acquisition recipe, except
for the added decayed stabilization terms:

- `result_policy_improvement_assignment_weight=10`
- `result_policy_entropy_weight=0.05`
- `result_policy_batch_diversity_weight=0.1`
- `result_policy_stabilization_decay_steps=1200`
- exact-grid natural `0..19`
- frozen product semantic decoder
- CLI seed `9`
- 1600 steps with checkpoint snapshots every 100 steps

Note: the saved config confirms CLI seed `9`, so this is a fresh-source
stabilization test, not a paired seed-7 ablation.

## Source Curve

| Step | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| `0` | `0.0200` | `0.0275` | `1.0000` | `0.0200` |
| `500` | `0.5750` | `0.0400` | `1.0000` | `0.5750` |
| `700` | `0.7050` | `0.0650` | `1.0000` | `0.7050` |
| `900` | `0.7050` | `0.0525` | `1.0000` | `0.7050` |
| `1200` | `0.5800` | `0.0400` | `1.0000` | `0.5800` |
| `1300` | `0.1900` | `0.0575` | `1.0000` | `0.1900` |
| `1600` | `0.2175` | `0.0300` | `1.0000` | `0.2175` |
| final eval | `0.1825` | `0.0703` | `1.0000` | `0.2266` |

The source peaked at `0.7050` before the stabilization decay completed, then
collapsed after the active source objective decayed to zero. The final metrics
record `final_result_policy_entropy_weight=0.0`,
`final_result_policy_batch_diversity_weight=0.0`, and
`final_result_policy_improvement_assignment_weight=0.0`.

## Decision

Label:

```text
source_acquisition_entropy_diversity_decay_negative
```

Do not run downstream additive handoff from this source family. Its best
source snapshots are below the recent fresh-source boundary, and the final
checkpoint is far weaker than the `src7` baseline final source (`0.8100`).
Running the full handoff/continuation/readout recipe here would mostly retest
weak-source transfer.

## Interpretation

Small entropy and batch-diversity terms did not rescue source acquisition.
They produced a temporary mid-run source at about `0.7050`, but the decay
schedule left no active source objective after step `1200`, and the policy
collapsed instead of preserving the learned calculator behavior.

The next source-acquisition variants should keep a nonzero floor or use an
anchor/selection objective if they use entropy/diversity. A pure decay-to-zero
stabilization schedule is a dead end for this recipe.

## Anti-Rerun Note

Do not repeat this exact source-only configuration:

```text
entropy=0.05, batch_diversity=0.1, improvement_assignment=10, decay_steps=1200,
answer_loss_weight=0, seed=9, 1600 steps
```

Next useful tests:

- keep a nonzero improvement-assignment or entropy/diversity floor;
- anchor the source policy before decaying auxiliary pressure;
- train source acquisition directly against a cheap proxy for 600-step handoff
  or continuation slope.

## Verification

The source-acquisition run completed and wrote `diagnostic_snapshots.csv` plus
`metrics.json`. No downstream handoff was run because the source curve failed
the acquisition gate.
