# Phase 7 Sixty-Sixth Task: Reduced Readout Budget Validation

## Status

Completed 2026-05-29.

## Question

Can the selected-source continuation recipe use fewer than 1600 stable-policy
readout-adaptation steps while keeping calculator-dependent non-bottleneck
accuracy above `0.90`?

## Setup

- Started from the continued selected-source frozen-policy checkpoints for
  `src4` step-1200/add2 and `src5` step-1100/add5.
- Loaded full model checkpoints.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Tested 200-step and 600-step readout budgets.

## Result

| Run | Readout steps | Final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4` selected continued | `200` | `0.8775` | `0.9025` at `200` | `0.8050` | `0.0000` | `0.0225` |
| `src4` selected continued | `600` | `0.9025` | `0.9250` at `500` | `0.8000` | `0.0025` | `0.0175` |
| `src5` selected continued | `200` | `0.9275` | `0.9325` at `100` | `0.8000` | `0.0000` | `0.0075` |
| `src5` selected continued | `600` | `0.9325` | `0.9525` at `600` | `0.8250` | `0.0000` | `0.0250` |

## Decision

```text
reduced_readout_budget_600_positive_200_mixed
```

The 200-step readout budget is not robust across selected sources. The 600-step
budget passes both tested selected lineages and cuts the stable readout stage
from 1600 to 600 steps.

## Next

Reduce the 600-step source-selection probe or the 800-step continuation stage,
or train source policies directly for early handoff and continuation slope.
