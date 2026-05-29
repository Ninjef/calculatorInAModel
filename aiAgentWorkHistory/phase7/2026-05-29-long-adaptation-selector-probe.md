# 2026-05-29 Long-Adaptation Selector Probe

## Aim

Check whether the `src5` source-accuracy-selected runner-up, step `1500`, beats
the 600-step handoff-probe-selected step `1100` after the same long stable-policy
adaptation.

This tests whether the earlier mixed result was because the 600-step frozen
handoff selector and the long-adaptation selector disagree on `src5`.

## Run

Run root:

```text
runs/2026-05-29_phase7_long_adaptation_selector_probe
```

Cell:

| Cell | Run directory |
| --- | --- |
| `src5_step1500_add5` | `runs/2026-05-29_phase7_long_adaptation_selector_probe/source_seed5_step1500_additive_seed5_freeze_policy_backbone_steps1600/2026-05-28_204104_527940_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

Configuration:

- Started from the existing `src5` step-1500 800-step frozen-policy additive
  handoff checkpoint.
- Loaded full model checkpoint.
- Used additive, non-bottleneck result-space calculator mode.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 1600 steps with snapshots every 100.

## Results

| Run | Frozen handoff final | Adapted final eval | Adapted best normal | Last inj-zero | Last forced-random | Last oracle | Last calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src5_step1100_selected` | `0.7950` | `0.9250` | `0.9325` at `1600` | `0.0000` | `0.0150` | `0.9525` | `0.8275` |
| `src5_step1500_runnerup` | `0.6975` | `0.9100` | `0.9400` at `1500` | `0.0000` | `0.0250` | `0.9525` | `0.9325` |
| `src5_old_finalsource` | `0.5550` | `0.9500` | `0.9625` at `1500` | `0.0025` | `0.0350` | `0.9050` | `0.8325` |

## Conclusion

For the fair `src5` runner-up comparison, the 600-step handoff-probe-selected
checkpoint still wins after long stable-policy adaptation: step `1100` reaches
`0.9250`, while step `1500` reaches `0.9100`.

The older final-source long-adaptation result remains better, but the gap is
not explained by the source-accuracy-selected step-1500 checkpoint being a
better long-adaptation candidate. Also, step `1500` ends with higher calculator
result accuracy but lower answer accuracy than step `1100`, reinforcing that
calculator action accuracy is not enough as a source selector.

Label:

```text
long_adaptation_selector_probe_step1500_negative
```

## Anti-Rerun Note

Do not repeat `src5` step-1500 800-step frozen handoff into no-anchor
`--freeze-calculator-policy-backbone`, LR `3e-4`, 1600-step adaptation as
novelty.

Next useful tests should compare the selected checkpoint against the exact old
final-source checkpoint lineage, or inspect what differed between the old
final-source frozen handoff and reproduced final-source/step checkpoints.
