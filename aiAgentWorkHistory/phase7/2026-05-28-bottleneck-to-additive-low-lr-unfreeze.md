# 2026-05-28 Bottleneck-to-Additive Low-LR Unfreeze

## Goal

Test whether the adapted non-bottleneck handoff can survive a simple
full-policy unfreeze at a lower learning rate.

## Anti-Rerun Check

The ledger allowed controlled unfreezing after the frozen handoff and longer
downstream adaptation results. This task changes the freeze state and LR; it is
not another frozen continuation.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_unfreeze_probe
```

Configuration:

- resumed from adapted weak-source additive checkpoints;
- `--semantic-decoder-checkpoint-load-scope full_model`;
- no `--freeze-calculator-policy`;
- global LR `3e-4`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

## Results

| Run | Final eval before | Final eval after | Best normal after | Last injection-zero | Last forced-random | Last oracle | Learned calc before | Learned calc after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` unfreeze | `0.6050` | `0.5200` | `0.6500` at `0` | `0.0225` | `0.1200` | `0.7100` | `0.8725` | `0.3000` |
| `src5_add5` unfreeze | `0.8175` | `0.8100` | `0.8325` at `0` | `0.0225` | `0.1125` | `0.7900` | `0.8000` | `0.2525` |

## Conclusion

Label:

```text
bottleneck_to_additive_low_lr_unfreeze_policy_collapse_negative
```

Even a ten-times smaller LR full unfreeze collapses learned calculator-result
accuracy. Normal accuracy can partly survive because the downstream path has
already adapted, but this is not a safe policy-preserving handoff.

## Next

- Do not repeat this exact low-LR full-unfreeze probe.
- Try selective unfreezing or explicit policy-retention regularization.
- Gate any unfreeze schedule by calculator-result accuracy, not only answer
  accuracy.

## Verification

No code changed in this task. Documentation formatting check:

```bash
git diff --check
```
