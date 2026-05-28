# 2026-05-28 Bottleneck-to-Additive Transfer Replication

## Goal

Replicate the frozen-policy bottleneck-to-additive handoff across source
bottleneck checkpoints and additive seeds without repeating the original
same-seed checkpoint cell as the only evidence.

## Periodic Review

Before running, I reviewed `CLAUDE.md`, `HYPOTHESIS_LEDGER.md`, and the Phase 7
fact sheet. The anti-rerun constraints ruled out:

- repeating the same frozen `src2_add2` handoff as novelty;
- repeating unfrozen compatible transfer, which already destroyed the policy;
- repeating direct non-bottleneck assignment or causal-gap runs.

The allowed next test was seed/checkpoint replication, staged unfreezing, or a
policy-acquisition change.

I attempted to spawn a sub-agent for checkpoint inventory, but the agent pool
reported that the thread limit was reached. I kept the checkpoint selection
local.

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

Cells:

| Cell | Source checkpoint | Additive CLI seed |
| --- | --- | ---: |
| `src2_add2` | previous strong source from seed-2 1600-step run | `2` |
| `src2_add4` | previous strong source from seed-2 1600-step run | `4` |
| `src4_add2` | seed-4 1600-step final checkpoint | `2` |
| `src4_add4` | seed-4 1600-step final checkpoint | `4` |
| `src5_add5` | seed-5 1600-step final checkpoint | `5` |

## Results

| Cell | Final eval | Best normal | Last injection-zero | Last forced-random | Last oracle | Step 0 learned calc | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src2_add2` | `0.9400` | `0.9475` at `800` | `0.0175` | `0.0500` | `0.9600` | `0.9125` | `0.9200` |
| `src2_add4` | `0.9525` | `0.9325` at `750` | `0.0200` | `0.0425` | `0.9600` | `0.9000` | `0.9150` |
| `src4_add2` | `0.3025` | `0.3150` at `800` | `0.0000` | `0.0375` | `0.3125` | `0.8650` | `0.8725` |
| `src4_add4` | `0.3375` | `0.3200` at `750` | `0.0000` | `0.0375` | `0.3075` | `0.8325` | `0.8575` |
| `src5_add5` | `0.5550` | `0.5725` at `800` | `0.0125` | `0.0425` | `0.5600` | `0.8475` | `0.8000` |

## Conclusion

Label:

```text
bottleneck_to_additive_freeze_policy_source_quality_mixed
```

The strong source checkpoint replicated across additive seeds and retained the
same causal calculator-use signature: normal/oracle high, injection-zero and
forced-random near chance, learned calculator-result accuracy high.

The weaker source checkpoints preserved learned calculator-result accuracy but
did not produce high downstream answer accuracy by step `800`. This suggests
that action accuracy alone is not a sufficient source-quality metric; the
source representation/output geometry and downstream adaptation dynamics also
matter.

## Next

- Do not repeat these exact 800-step frozen matrix cells as novelty.
- Test source checkpoint selection metrics, especially best snapshots or
  calibration/readout geometry rather than final action accuracy alone.
- Try stronger downstream readout adaptation or controlled unfreezing that
  preserves the learned calculator policy.

## Verification

No code changed in this task. The previous code-change commit passed:

```text
106 passed
```
