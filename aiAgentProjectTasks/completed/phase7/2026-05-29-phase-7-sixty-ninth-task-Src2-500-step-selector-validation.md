# Phase 7 Sixty-Ninth Task: `src2` 500-Step Selector Validation

## Status

Completed 2026-05-29.

## Question

Does the shortened 500-step handoff selector also work on the `src2`
source-accuracy counterexample?

## Setup

- Reused existing additive seed-4 frozen-policy handoff traces.
- Compared source step `1300` against final / step `1600`.
- Audited normal accuracy at 400, 500, 600, and 800 handoff steps.

## Result

| Candidate | Source normal/calc | Normal @ 400 | Normal @ 500 | Normal @ 600 | Final handoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `1300` | `0.9475` | `0.4100` | `0.5875` | `0.7200` | `0.8675` |
| final / step `1600` | `0.9150` | `0.5700` | `0.6900` | `0.8025` | `0.9525` |

## Decision

```text
src2_500_step_selector_validation_positive
```

The 500-step selector picks the known better final checkpoint on `src2`, despite
that checkpoint having lower source normal/calculator accuracy than step
`1300`.

## Next

Validate 500-step selection on newly acquired source checkpoints, or train
source policies directly for early handoff and continuation slope.
