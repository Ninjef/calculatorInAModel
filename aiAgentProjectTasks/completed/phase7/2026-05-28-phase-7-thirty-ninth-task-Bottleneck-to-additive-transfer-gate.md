# Phase 7 Thirty-Ninth Task: Bottleneck-to-Additive Transfer Gate

## Status

Completed on 2026-05-28.

## Question

Can a result policy trained in the answer-decoder bottleneck be loaded into an
additive non-bottleneck model, and can answer-only downstream training learn to
depend on that calculator path rather than the bypass?

## Implementation

- Added `compatible_model` checkpoint loading so a bottleneck checkpoint can
  initialize matching tensors in an additive model.
- Added `--freeze-calculator-policy` to freeze the embeddings, pre-hook block,
  and result action head while leaving the additive output projection and
  downstream answer path trainable.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_gate
```

Source checkpoint:

```text
runs/2026-05-28_phase7_hard_improvement_assignment_convergence_gate/answer0_w10_steps1600/2026-05-28_164332_598334_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

## Result

| Setup | Final eval exact | Best normal | Last injection-zero | Last forced-random | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compatible load, no freeze | `0.7825` | `0.8075` | `0.7675` | `0.7375` | `0.7825` | `0.0250` |
| compatible load, freeze policy | `0.9400` | `0.9475` | `0.0175` | `0.0500` | `0.9600` | `0.9200` |

The unfrozen run kept the transferred result policy only at step `0`
(`0.9125` learned calc) and collapsed it by step `50` (`0.0300`). The frozen
run preserved the policy (`0.8925` at step `50`, `0.9200` at step `800`) and
made the additive answer path causally depend on the calculator.

## Decision

```text
bottleneck_to_additive_freeze_policy_handoff_partial_positive
```

This is a real non-bottleneck calculator-dependence result, but not the final
goal: it is staged, inherits a bottleneck-trained forced-assignment policy, and
freezes that policy during handoff.

## Next

- Replicate across bottleneck checkpoints and additive seeds.
- Test staged unfreezing schedules.
- Search for cheaper or less prescriptive policy acquisition before handoff.
