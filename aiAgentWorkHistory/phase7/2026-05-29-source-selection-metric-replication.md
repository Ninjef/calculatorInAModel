# 2026-05-29 Source Selection Metric Replication

## Goal

Replicate the source-checkpoint selection idea on the strong `src2` source and
test whether selecting by bottleneck source accuracy alone is a reliable
handoff metric.

## Periodic Review

The previous source-selection gate showed that selecting the `src5` step-1500
source snapshot improved immediate frozen-policy handoff versus the unstable
final checkpoint. The ledger explicitly warned that source action accuracy was
useful but not sufficient. This run therefore tested a non-duplicative
question: whether the same source-normal-accuracy selector improves, matches,
or harms a source that already transfers strongly.

## Runs

Run root:

```text
runs/2026-05-29_phase7_source_checkpoint_selection_replication
```

Source reproduction:

```text
source_seed2_snapshots_steps1600
```

Configuration:

- bottleneck source, `calculator_bottleneck_mode=answer_decoder`;
- `calculator_estimator=direct_feedback_alignment`;
- `calculator_action_head=result_space`;
- frozen product semantic decoder;
- `result_policy_improvement_assignment_weight=10`;
- exact-grid natural `0..19`;
- CLI seed `2`;
- 1600 steps;
- `--checkpoint-every 100`.

Transfer cells:

| Cell | Source checkpoint | Additive CLI seed |
| --- | --- | ---: |
| selected-source transfer | source step `1300` | `4` |
| final-source control | reproduced source final | `4` |

Both transfer cells used the same additive configuration:

- `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible checkpoint load from the bottleneck source;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 800 steps.

## Results

Source checkpoints:

| Source checkpoint | Source normal | Source learned calc | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `1300` | `0.9475` | `0.9475` | `0.0300` | `0.0225` | `1.0000` |
| step `1600` / final | `0.9150` | `0.9150` | `0.0375` | `0.0250` | `1.0000` |

Additive transfers:

| Setup | Source checkpoint | Source normal | Additive final | Best normal | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-source handoff | step `1300` | `0.9475` | `0.8675` | `0.8875` at `800` | `0.0400` | `0.0450` | `0.8875` | `0.9250` |
| final-source control | final / step `1600` | `0.9150` | `0.9525` | `0.9325` at `800` | `0.0200` | `0.0425` | `0.9600` | `0.9150` |

The final-source control matched the prior `src2_add4` baseline (`0.9525`),
so the selected-source drop was not caused by reproducing the source run.

## Conclusion

Label:

```text
bottleneck_to_additive_source_accuracy_selector_negative
```

Source checkpoint selection remains important, but selecting by source
normal/calculator accuracy alone is now disproven. On `src2`, the best
source-accuracy checkpoint (`0.9475`) transferred substantially worse than the
lower-accuracy final checkpoint (`0.9150`): `0.8675` versus `0.9525` final
additive eval.

This suggests the missing source-quality metric is handoff geometry, not just
calculator-result correctness. Good next metrics should look at how readily the
post-hook additive readout can learn from the frozen calculator output, for
example early transfer slope, oracle-vs-normal gap after short adaptation, or a
linear/readout probe on the frozen source state.

## Anti-Regression Note

Do not repeat `src2` step-1300 versus final additive seed-4 frozen-policy
800-step transfer as novelty. Next useful tests are:

- source-quality probes that predict handoff before a full 800-step transfer;
- source acquisition that optimizes handoff-friendly geometry;
- selected-source replication only if the selector is no longer just source
  normal/calculator accuracy.

## Verification

No code changed in this task. The source reproduction, selected-source
transfer, and final-source control all completed and wrote metrics under the
run root above.
