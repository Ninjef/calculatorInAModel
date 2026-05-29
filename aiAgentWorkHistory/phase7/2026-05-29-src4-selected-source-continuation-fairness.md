# 2026-05-29 `src4` Selected-Source Continuation Fairness

## Aim

Apply the same fair continuation recipe from the `src5` selected-source result
to the weaker `src4` step-1200 selected source.

This tests whether selected-source handoff plus an extra frozen-policy
continuation can further improve the previously weak `src4` transfer before
stable-policy readout adaptation.

## Runs

Run root:

```text
runs/2026-05-29_phase7_src4_selected_source_continuation_fairness
```

Cells:

| Cell | Run directory |
| --- | --- |
| `src4_step1200_add2_continue800` | `runs/2026-05-29_phase7_src4_selected_source_continuation_fairness/source_seed4_step1200_additive_seed2_continue800_freeze_policy/2026-05-28_211114_191728_model-c-op0-19-fullgrid/model-c-2digit-seed6` |
| `src4_step1200_add2_continued_long1600` | `runs/2026-05-29_phase7_src4_selected_source_continuation_fairness/source_seed4_step1200_additive_seed2_continued_freeze_policy_backbone_steps1600/2026-05-28_211603_439040_model-c-op0-19-fullgrid/model-c-2digit-seed6` |

Configurations:

- Continuation leg:
  - Started from `src4` step-1200 800-step frozen-policy additive handoff.
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
| old `src4` final-source lineage | n/a | `0.6050` | `0.7550` | `0.7600` at `1100` | `0.0525` | `0.0900` | `0.7325` | `0.8550` |
| selected `src4` step-1200 direct long | `0.7800` | n/a | `0.8900` | `0.8975` at `1400` | `0.0000` | `0.0175` | `0.8675` | `0.8225` |
| selected `src4` step-1200 continued | `0.7800` | `0.8150` | `0.9125` | `0.9475` at `1400` | `0.0075` | `0.0250` | `0.8825` | `0.8025` |

For comparison, the selected `src5` fair continuation chain reached `0.9425`
long final eval with `0.0000` injection-zero and `0.0225` forced-random.

## Conclusion

The fair continuation recipe also improves `src4`: selected step-1200 moves
from `0.7800` after the first frozen-policy handoff to `0.8150` after the extra
frozen continuation, then reaches `0.9125` under stable-policy readout
adaptation. This beats both the old `src4` final-source long adaptation
(`0.7550`) and the direct selected-source long adaptation (`0.8900`).

The gain is smaller than for `src5`, and final controls are slightly less clean
than the direct selected run (`injection-zero 0.0075` versus `0.0000`), but
forced-random remains near chance (`0.0250`). The current scalable-ish recipe is
therefore: use the 600-step handoff probe for source selection, continue the
frozen-policy additive handoff, then adapt the readout with the policy backbone
frozen.

Label:

```text
src4_selected_source_continuation_fairness_positive
```

## Anti-Rerun Note

Do not repeat `src4` step-1200 selected 800-step frozen handoff plus another
800-step frozen-policy continuation plus no-anchor policy-backbone-frozen
1600-step adaptation as novelty.

Next useful tests should reduce the expensive handoff-probe/continuation cost,
or move source acquisition toward optimizing early handoff and continuation
slope directly.
