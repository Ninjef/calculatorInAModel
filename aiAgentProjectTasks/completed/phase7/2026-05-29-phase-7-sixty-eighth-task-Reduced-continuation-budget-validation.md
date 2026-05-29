# Phase 7 Sixty-Eighth Task: Reduced Continuation Budget Validation

## Status

Completed 2026-05-29.

## Question

Can the extra frozen-policy continuation stage be reduced from 800 to 600 steps
while keeping the reduced 600-step stable readout stage?

## Setup

- Started from the selected 800-step frozen-policy handoff checkpoints for
  `src4` step-1200/add2 and `src5` step-1100/add5.
- Ran 600-step frozen-policy continuation with LR `3e-3`.
- Then ran 600-step no-anchor stable readout adaptation with
  `--freeze-calculator-policy-backbone` and LR `3e-4`.
- Used additive, non-bottleneck result-space calculator mode throughout.
- Used exact-grid natural `0..19`, answer loss weight `1`, and frozen semantic
  decoder.

## Result

| Run | Continuation steps | Readout steps | Continuation final | Readout final | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4` reduced | `600` | `600` | `0.7950` | `0.8750` | `0.8975` at `500` | `0.8000` | `0.0000` | `0.0200` |
| `src4` reference | `800` | `600` | `0.8150` | `0.9025` | `0.9250` at `500` | `0.8000` | `0.0025` | `0.0175` |
| `src5` reduced | `600` | `600` | `0.8850` | `0.9275` | `0.9525` at `600` | `0.8250` | `0.0000` | `0.0275` |
| `src5` reference | `800` | `600` | `0.8800` | `0.9325` | `0.9525` at `600` | `0.8250` | `0.0000` | `0.0250` |

## Decision

```text
reduced_continuation_budget_600_source_sensitive_negative
```

The 600-step continuation works for `src5` but fails the `0.90` final-eval gate
for weaker `src4`. The current validated recipe should keep 800 continuation
steps for weak selected sources.

## Next

Keep 800 continuation for weak sources, test an intermediate 700-step
continuation only if fine-grained cost tuning matters, or train source policies
for better continuation slope directly.
