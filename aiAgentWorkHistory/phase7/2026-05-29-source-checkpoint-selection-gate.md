# 2026-05-29 Source Checkpoint Selection Gate

## Goal

Test whether selecting a better bottleneck source snapshot improves
bottleneck-to-additive handoff, instead of repeating downstream rescue from the
same final weak-source checkpoint.

## Periodic Review

Before running, I reviewed `CLAUDE.md`, `HYPOTHESIS_LEDGER.md`, the Phase 7
fact sheet, and the recent bottleneck-to-additive work history. The anti-rerun
constraints ruled out:

- repeating the original 800-step frozen-policy transfer matrix as novelty;
- repeating 400-step policy-backbone freeze or tiny-anchor variants;
- repeating 1600-step stable-policy downstream adaptation as novelty.

The allowed direction was source-policy acquisition or source checkpoint
selection. The existing `src5` bottleneck source was a good target because its
source diagnostic peaked at step `1500` (`0.9200`) but the final checkpoint fell
to `0.8200`, and the old additive transfer used the final checkpoint.

## Runs

Run root:

```text
runs/2026-05-29_phase7_source_checkpoint_selection_gate
```

Source acquisition cell:

```text
source_seed5_snapshots_steps1600
```

Configuration:

- bottleneck source, `calculator_bottleneck_mode=answer_decoder`;
- `calculator_estimator=direct_feedback_alignment`;
- `calculator_action_head=result_space`;
- frozen product semantic decoder;
- `result_policy_improvement_assignment_weight=10`;
- exact-grid natural `0..19`;
- CLI seed `5`;
- 1600 steps;
- `--checkpoint-every 100`.

Selected source checkpoint:

```text
runs/2026-05-29_phase7_source_checkpoint_selection_gate/source_seed5_snapshots_steps1600/2026-05-28_193325_798612_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed7/checkpoint_snapshots/step_01500_weights.pt
```

Additive handoff cell:

```text
source_seed5_step1500_additive_seed5_freeze_policy_steps800
```

Configuration:

- additive path, `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible checkpoint load from the selected bottleneck snapshot;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- CLI seed `5`;
- 800 steps.

## Results

Source snapshots:

| Source checkpoint | Source normal | Source learned calc | Injection-zero | Oracle |
| --- | ---: | ---: | ---: | ---: |
| step `1400` | `0.8900` | `0.8900` | `0.0675` | `1.0000` |
| step `1500` | `0.9200` | `0.9200` | `0.0450` | `1.0000` |
| step `1600` / final | `0.8325` | `0.8325` | `0.0575` | `1.0000` |

Additive transfer:

| Setup | Source checkpoint | Final eval | Best normal | Last injection-zero | Last forced-random | Last oracle | Last learned calc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old `src5_add5` frozen-policy baseline | final source | `0.5550` | `0.5725` at `800` | `0.0125` | `0.0425` | `0.5600` | `0.8000` |
| selected-snapshot frozen-policy transfer | step `1500` source | `0.6975` | `0.6925` at `800` | `0.0150` | `0.0350` | `0.7025` | `0.9000` |

The selected source snapshot improved the immediate 800-step frozen handoff by
`+0.1425` absolute over the old final-checkpoint baseline while preserving the
calculator-use causal signature: injection-zero stayed near chance and oracle
tracked normal accuracy.

## Conclusion

Label:

```text
bottleneck_to_additive_source_checkpoint_selection_partial
```

Source checkpoint selection matters. A source snapshot with higher bottleneck
calculator-result accuracy produced a materially better additive handoff than
the unstable final checkpoint.

This is not a full fix. The selected snapshot had source normal/calculator
accuracy `0.9200`, yet its immediate additive handoff reached only `0.6975`,
well below the strong `src2` handoff (`0.9400-0.9525`) and below later `src5`
stable-policy adaptation (`0.9500`). Source action accuracy is therefore a
useful selection metric but not sufficient; the source representation/readout
geometry still controls how quickly the additive path can use the calculator.

## Anti-Regression Note

Do not repeat the same `src5` step-1500 selected-snapshot frozen-policy
800-step handoff as novelty. Next useful tests are:

- source-selection metrics beyond source normal accuracy;
- acquiring source checkpoints with handoff-friendly representation geometry;
- directly transferring strong selected snapshots across more additive seeds;
- utility-aware downstream/readout adaptation under stable calculator use.

## Verification

No code changed in this task. The new source and transfer runs completed and
wrote metrics under the run root above.
