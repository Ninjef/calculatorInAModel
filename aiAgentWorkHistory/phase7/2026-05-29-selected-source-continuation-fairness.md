# 2026-05-29 Selected-Source Continuation Fairness

## Aim

Test whether the old `src5` final-source long-adaptation advantage came from a
special final-source checkpoint, or from receiving an extra 800-step frozen-policy
continuation before the 1600-step stable-policy adaptation.

## Runs

Run root:

```text
runs/2026-05-29_phase7_selected_source_continuation_fairness
```

Cells:

| Cell | Run directory |
| --- | --- |
| `src5_step1100_add5_continue800` | `runs/2026-05-29_phase7_selected_source_continuation_fairness/source_seed5_step1100_additive_seed5_continue800_freeze_policy/2026-05-28_210232_787778_model-c-op0-19-fullgrid/model-c-2digit-seed9` |
| `src5_step1100_add5_continued_long1600` | `runs/2026-05-29_phase7_selected_source_continuation_fairness/source_seed5_step1100_additive_seed5_continued_freeze_policy_backbone_steps1600/2026-05-28_210408_648304_model-c-op0-19-fullgrid/model-c-2digit-seed9` |

Configurations:

- Continuation leg:
  - Started from `src5` step-1100 800-step frozen-policy additive handoff.
  - Loaded full model checkpoint.
  - Used `--freeze-semantic-decoder`.
  - Used `--freeze-calculator-policy`.
  - LR `3e-3`, answer loss weight `1`, exact-grid natural `0..19`.
  - Ran 800 steps with snapshots every 100.
- Long adaptation leg:
  - Started from the continued checkpoint above.
  - Loaded full model checkpoint.
  - Used `--freeze-semantic-decoder`.
  - Used `--freeze-calculator-policy-backbone`.
  - Used no result-policy anchor.
  - LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
  - Ran 1600 steps with snapshots every 100.

## Results

| Run | Start final | Continued final | Long final eval | Long best normal | Long inj-zero | Long forced-random | Long oracle | Long calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `src5` final-source lineage | `0.5550` | `0.8175` | `0.9500` | `0.9625` at `1500` | `0.0025` | `0.0350` | `0.9050` | `0.8325` |
| selected `src5` step-1100 lineage | `0.7950` | `0.8800` | `0.9425` | `0.9625` at `800` | `0.0000` | `0.0225` | `0.9600` | `0.8275` |

Related direct long-adaptation controls:

| Run | Direct 800-step frozen handoff | Direct long final eval |
| --- | ---: | ---: |
| `src5` step-1100 selected | `0.7950` | `0.9250` |
| `src5` step-1500 runner-up | `0.6975` | `0.9100` |

## Conclusion

The old final-source advantage was mostly a continuation-depth fairness issue,
not evidence that the old final-source checkpoint was intrinsically better for
stable-policy readout adaptation. Giving the probe-selected step-1100 lineage
the same extra frozen-policy continuation raised its handoff from `0.7950` to
`0.8800`, then its long stable-policy adaptation reached `0.9425`, nearly
matching the old final-source `0.9500`.

The selected lineage also remained calculator-dependent under controls:
injection-zero stayed `0.0000`, and forced-random stayed near chance at
`0.0225`.

Label:

```text
selected_source_continuation_fairness_positive
```

## Anti-Rerun Note

Do not repeat `src5` step-1100 selected 800-step frozen handoff plus another
800-step frozen-policy continuation plus no-anchor policy-backbone-frozen
1600-step adaptation as novelty.

Next useful tests should apply the same fair continuation recipe to `src4`
step-1200, or move from post-hoc source selection toward source acquisition
that optimizes early handoff and continuation slope.
