# 2026-05-28 Bottleneck-to-Additive Downstream Adaptation

## Goal

Disentangle weak-source frozen handoff failure from slow downstream adaptation.
The previous replication matrix showed that weaker source checkpoints retained
calculator action accuracy but reached low answer accuracy by 800 additive
steps. This task asks whether another 800 answer-loss steps lets the downstream
path catch up while the calculator policy remains frozen.

## Anti-Rerun Check

This is not a repeat of the 800-step frozen matrix. It resumes two weak-source
additive final checkpoints and changes the adaptation budget. The ledger
allowed longer or stronger downstream adaptation after the mixed replication
result.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation
```

Configuration:

- resumed from additive final weights with
  `--semantic-decoder-checkpoint-load-scope full_model`;
- `--freeze-calculator-policy`;
- `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- answer loss weight `1`;
- no assignment target;
- exact-grid natural `0..19`;
- 800 additional steps.

## Results

| Run | Final eval before | Final eval after | Best normal after | Last injection-zero | Last forced-random | Last oracle | Step 0 learned calc | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` continue | `0.3025` | `0.6050` | `0.5725` at `550` | `0.0025` | `0.0625` | `0.5725` | `0.8650` | `0.8725` |
| `src5_add5` continue | `0.5550` | `0.8175` | `0.8150` at `800` | `0.0000` | `0.0425` | `0.8075` | `0.8475` | `0.8000` |

## Conclusion

Label:

```text
bottleneck_to_additive_longer_downstream_adaptation_partial
```

Weak-source frozen handoffs can improve substantially with more downstream
answer-loss training, and the causal calculator-use signature remains intact.
This weakens the idea that weaker sources are unusable. However, even after a
total 1600-step additive adaptation budget, the weak-source continuations stay
below the strong-source handoffs near `0.95` final eval.

## Next

- Source checkpoint selection should consider downstream handoff quality, not
  only frozen result-policy accuracy.
- Test controlled unfreezing or stronger readout adaptation.
- Do not spend another task on the same one-extra-800-step continuation.

## Verification

No code changed in this task. Documentation formatting check:

```bash
git diff --check
```
