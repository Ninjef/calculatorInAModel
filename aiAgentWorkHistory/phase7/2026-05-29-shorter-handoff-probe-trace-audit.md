# 2026-05-29 Shorter Handoff Probe Trace Audit

## Aim

Reduce the source-selection probe cost in the current selected-source recipe.

The prior positive selector used 600-step frozen-policy additive handoff probes
to choose source checkpoints. Existing probe traces include 100-step snapshots,
so this audit checks whether an earlier snapshot would have selected the same
source checkpoints without rerunning the probes.

## Source Traces

Run roots:

```text
runs/2026-05-29_phase7_handoff_probe_selector_validation
runs/2026-05-29_phase7_handoff_probe_selector_src4
runs/2026-05-29_phase7_source_checkpoint_selection_gate
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_replication
```

## Results

`src5` additive seed `5` candidate traces:

| Source checkpoint | Normal @ 400 | Normal @ 500 | Normal @ 600 | Final handoff |
| --- | ---: | ---: | ---: | ---: |
| step `1100` | `0.3475` | `0.5575` | `0.6825` | `0.6850` probe / `0.7950` full |
| step `1400` | `0.4150` | `0.5350` | `0.6625` | `0.6400` |
| step `1500` | `0.4325` | `0.4600` | `0.5250` | `0.6975` full |
| final / step `1600` | `0.2975` | `0.4100` | `0.4525` | `0.5550` full |

`src4` additive seed `2` candidate traces:

| Source checkpoint | Normal @ 400 | Normal @ 500 | Normal @ 600 | Final handoff |
| --- | ---: | ---: | ---: | ---: |
| step `1000` | `0.3875` | `0.4275` | `0.5450` | `0.5225` |
| step `1200` | `0.5450` | `0.5875` | `0.6250` | `0.6425` probe / `0.7800` full |
| final / step `1600` | `0.2400` | `0.2325` | `0.2675` | `0.3025` full |

## Conclusion

The 400-step probe is still too short for `src5`: it would select step `1500`
instead of the eventual step-`1100` winner. By 500 steps, the probe selects the
same checkpoints as the 600-step selector for both audited source families:

- `src5`: step `1100`.
- `src4`: step `1200`.

This reduces the source-selection probe from 600 to 500 steps for the currently
validated candidate sets, a 16.7% cut in that stage. It is only a trace audit,
not a proof that 500 steps is universally safe on new sources.

Label:

```text
shorter_handoff_probe_500_trace_positive_400_negative
```

## Anti-Rerun Note

Do not repeat this same trace audit over the existing `src5`
`1100/1400/1500/final` and `src4` `1000/1200/final` candidate traces as
novelty.

Next useful tests should validate the 500-step selector on newly acquired
source checkpoints, reduce the 800-step frozen-policy continuation, or optimize
source acquisition for early handoff/continuation slope directly.
