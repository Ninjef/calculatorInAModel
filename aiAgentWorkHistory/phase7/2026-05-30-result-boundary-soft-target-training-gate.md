# 2026-05-30 Result-Boundary Soft-Target Training Gate

## Question

Can a softer answer-derived result-boundary target train the result-space
calculator policy better than hard-best result CE, by tolerating uncertainty
instead of requiring exact argmin selection?

This directly followed the proposal/adaptive diagnostics, which showed that
static candidate selection can miss the exact full-enum best. The softer target
construction is meant to answer whether exact argmin recovery is really needed.

## Setup

Matched the known full-grid upstream-open result-boundary source gate:

- full `0..19` ordered-pair grid (`400` prompts)
- frozen product answer decoder
- upstream open
- `result_space` calculator head
- `operand_spans` read position
- `answer_loss_weight=0`
- `result_boundary_target_loss_weight=1`
- `steps=200`
- `lr=0.01`
- `upstream_lr=0.0003`
- snapshots/checkpoints every `50`

Probe runs first measured target sharpness:

```text
runs/2026-05-30_phase7_result_boundary_soft_target_probe/t1
runs/2026-05-30_phase7_result_boundary_soft_target_probe/t4
runs/2026-05-30_phase7_result_boundary_soft_target_probe/t16
```

Training gates:

```text
runs/2026-05-30_phase7_result_boundary_soft_target_training/hard_matched_step200
runs/2026-05-30_phase7_result_boundary_soft_target_training/t1_step200
runs/2026-05-30_phase7_result_boundary_soft_target_training/t4_step200
```

## Target Sharpness

| Target | Temperature | True-result target mass | Effective results |
| --- | ---: | ---: | ---: |
| hard-best | `0.25` | `0.9999` | `1.001` |
| soft | `1.0` | `0.8003` | `2.722` |
| soft | `4.0` | `0.1336` | `28.347` |
| soft | `16.0` | `0.0412` | `38.164` |

Temperature `1.0` was the useful moderately soft setting. Temperature `4.0`
was included as a broad-target stress. Temperature `16.0` was essentially
uniform and was not trained.

## Results

Matched 200-step training:

| Target | Step-200 learned calc | Step-200 snapshot normal | Final eval |
| --- | ---: | ---: | ---: |
| hard-best `t=0.25` | `0.5450` | `0.5925` | `0.5475` |
| soft `t=1.0` | `0.2900` | `0.2775` | `0.2775` |
| soft `t=4.0` | `0.1350` | `0.1325` | `0.1275` |

For reference, the archival hard-best full-grid run also had step-200 learned
calculator-result accuracy `0.5450` and later reached `0.9675` at step `800`.
The matched hard run reproduced the step-200 point, so the soft-target miss is
not explained by a setup mismatch.

## Interpretation

Softer result-boundary targets do not improve the source-acquisition gate.

- Moderate softening (`t=1`) creates a real set-like target, with about `2.7`
  effective result classes, but it learns much more slowly than hard-best.
- Broad softening (`t=4`) dilutes the target too much and barely learns.
- This weakens the idea that the next result-boundary fix is a simple
  temperature/soft-target variant over the same full-enum loss table.

## Decision

```text
result_boundary_soft_target_training_negative
```

Do not continue static `soft_result` temperature ladders as novelty. A useful
next target-construction change needs to be more structural: set targets tied
to candidate uncertainty, streaming/evolving-checkpoint validation, or a
proposal model that reduces enumeration without simply softening the exact-grid
teacher.
