# Phase 7 Sixty-Second Task: Probe-Selected Policy-Backbone Adaptation

## Status

Completed 2026-05-29.

## Question

Do source checkpoints selected by the 600-step additive handoff probe reduce the
need for later anchoring or long downstream adaptation?

## Setup

- Continued from the probe-selected frozen-policy additive handoff checkpoints.
- Loaded full model checkpoints.
- Used additive, non-bottleneck result-space calculator mode.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- Used LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 1600 steps with snapshots every 100 steps.

## Result

| Run | Frozen handoff final | Adapted final eval | Best normal | Final calc | Final injection-zero | Final forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4` step-1200 selected | `0.7800` | `0.8900` | `0.8975` at `1400` | `0.8225` | `0.0000` | `0.0175` |
| `src5` step-1100 selected | `0.7950` | `0.9250` | `0.9325` at `1600` | `0.8275` | `0.0000` | `0.0150` |

Against older final-source long adaptation:

| Seed pair | Selected-source adapted final | Old final-source adapted final |
| --- | ---: | ---: |
| `src4/add2` | `0.8900` | `0.7550` |
| `src5/add5` | `0.9250` | `0.9500` |

## Decision

```text
probe_selected_policy_backbone_adaptation_mixed_positive
```

Probe selection strongly improves the weak `src4` downstream adaptation case,
but it does not universally dominate for long stable-policy adaptation.

## Next

Add a second-stage selector or acquisition objective for long-readout
adaptability, rather than assuming the 600-step frozen handoff probe is the
only source-quality metric needed downstream.
