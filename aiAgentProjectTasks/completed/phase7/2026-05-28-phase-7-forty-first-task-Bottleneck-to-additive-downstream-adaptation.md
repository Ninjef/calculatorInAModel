# Phase 7 Forty-First Task: Bottleneck-to-Additive Downstream Adaptation

## Status

Completed on 2026-05-28.

## Question

Can weak-source frozen bottleneck-to-additive handoffs catch up if the additive
downstream/readout path receives more answer-loss optimization while the
calculator policy stays frozen?

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_downstream_adaptation
```

Configuration:

- resumed from the additive final weights of weak-source frozen handoff cells;
- `--semantic-decoder-checkpoint-load-scope full_model`;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 800 additional steps.

## Result

| Run | Final eval before | Final eval after | Best normal after | Last injection-zero | Last oracle | Last learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` continue | `0.3025` | `0.6050` | `0.5725` | `0.0025` | `0.5725` | `0.8725` |
| `src5_add5` continue | `0.5550` | `0.8175` | `0.8150` | `0.0000` | `0.8075` | `0.8000` |

## Decision

```text
bottleneck_to_additive_longer_downstream_adaptation_partial
```

Longer downstream adaptation helps and preserves causal calculator dependence,
but it does not fully erase source checkpoint sensitivity.

## Next

- Do not repeat one more identical 800-step frozen continuation as novelty.
- Test source checkpoint selection metrics.
- Try stronger downstream adaptation or controlled unfreezing that preserves
  the calculator policy.
