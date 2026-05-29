# Phase 7 Fifty-First Task: Bottleneck-to-Additive Continuous Anchor Gate

## Status

Completed 2026-05-29.

## Question

Can the result-policy anchor use a continuous behavior gate so retention force
scales with calculator-accuracy shortfall instead of jumping discretely?

## Setup

- Added `--result-policy-anchor-gate-mode linear`.
- Added `--result-policy-anchor-gate-band`.
- Continued from the adapted weak-source frozen-policy handoff checkpoints.
- Loaded full model checkpoints and unfroze the policy.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Used base KL result-policy anchor `0.01`.
- Used `current_argmax_accuracy` threshold `0.85`, band `0.10`, max/gate
  weight `0.1`.
- Ran 400 steps with snapshots every 50 steps.

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` linear gate | `0.8375` | `0.8375` at `400` | `0.7675` | `0.0575` | `9/9` | `0.0385` |
| `src5_add5` linear gate | `0.9725` | `0.9525` at `400` | `0.7600` | `0.0000` | `9/9` | `0.0833` |

## Decision

```text
bottleneck_to_additive_continuous_anchor_gate_partial
```

Continuous gating is useful and lowered average retention weight. It slightly
beat fixed anchor `0.1` for `src4_add2`, but did not beat fixed/discrete gates
for `src5_add5`, so it is not a clean replacement recipe.

## Next

Avoid simple band sweeps. Next retention work should combine calculator
accuracy with answer utility, change which policy-path parameters move, or
improve source-policy acquisition.
