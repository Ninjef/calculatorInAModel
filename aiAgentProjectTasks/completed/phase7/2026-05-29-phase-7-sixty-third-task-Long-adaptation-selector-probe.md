# Phase 7 Sixty-Third Task: Long-Adaptation Selector Probe

## Status

Completed 2026-05-29.

## Question

Does the `src5` step-1500 checkpoint, which had higher source accuracy and was
the earlier source-accuracy-selected handoff, beat the handoff-probe-selected
step-1100 checkpoint after the same 1600-step stable-policy adaptation?

## Setup

- Continued from the existing `src5` step-1500 800-step frozen-policy additive
  handoff checkpoint.
- Loaded full model checkpoint.
- Used additive, non-bottleneck result-space calculator mode.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 1600 steps with snapshots every 100 steps.

## Result

| Run | Frozen handoff final | Adapted final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src5` step-1100 selected | `0.7950` | `0.9250` | `0.9325` at `1600` | `0.8275` | `0.0000` | `0.0150` |
| `src5` step-1500 runner-up | `0.6975` | `0.9100` | `0.9400` at `1500` | `0.9325` | `0.0000` | `0.0250` |

## Decision

```text
long_adaptation_selector_probe_step1500_negative
```

The source-accuracy-selected step-1500 checkpoint does not beat the 600-step
handoff-probe-selected step-1100 checkpoint under long stable-policy
adaptation.

## Next

Compare the selected checkpoint against the exact old final-source lineage, or
inspect why the older final-source long-adaptation run reached `0.9500`.
