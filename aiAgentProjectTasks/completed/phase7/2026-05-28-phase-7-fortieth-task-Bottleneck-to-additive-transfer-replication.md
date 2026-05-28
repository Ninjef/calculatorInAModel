# Phase 7 Fortieth Task: Bottleneck-to-Additive Transfer Replication

## Status

Completed on 2026-05-28.

## Question

Does the frozen-policy bottleneck-to-additive handoff replicate across additive
seeds and source bottleneck checkpoints?

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_replication
```

Configuration:

- additive path: `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible checkpoint load from bottleneck hard-assignment checkpoints;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- no assignment target;
- exact-grid natural `0..19`;
- 800 steps.

## Result

| Cell | Final eval | Best normal | Last injection-zero | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src2_add2` | `0.9400` | `0.9475` | `0.0175` | `0.9600` | `0.9200` |
| `src2_add4` | `0.9525` | `0.9325` | `0.0200` | `0.9600` | `0.9150` |
| `src4_add2` | `0.3025` | `0.3150` | `0.0000` | `0.3125` | `0.8725` |
| `src4_add4` | `0.3375` | `0.3200` | `0.0000` | `0.3075` | `0.8575` |
| `src5_add5` | `0.5550` | `0.5725` | `0.0125` | `0.5600` | `0.8000` |

## Decision

```text
bottleneck_to_additive_freeze_policy_source_quality_mixed
```

The strong seed-2 source replicated across additive seeds. The weaker seed-4
and seed-5 source checkpoints preserved calculator-result action accuracy but
did not give high downstream answer accuracy by step `800`.

## Next

- Select source checkpoints by more than final result-policy accuracy.
- Try stronger downstream/readout adaptation for weaker sources.
- Test controlled unfreezing that preserves the learned action policy.
