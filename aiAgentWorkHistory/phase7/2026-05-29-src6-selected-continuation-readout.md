# 2026-05-29 `src6` Selected Continuation and Readout

## Aim

Test whether the selected-source continuation/readout recipe generalizes to the
fresh `src6` source family after the 600-step selector chose the final source
checkpoint.

The prior `src6` handoff was a useful near miss: final-source 800-step frozen
handoff reached `0.8975`, just below the `0.90` non-bottleneck gate. This task
checks whether the established recipe clears the gate on this fresh source.

## Runs

Run root:

```text
runs/2026-05-29_phase7_src6_selected_continuation_readout
```

Continuation cell:

```text
source_seed6_final_additive_seed6_continue800_freeze_policy
```

Configuration:

- Started from the fresh `src6` final-source 800-step frozen-policy additive
  handoff.
- Loaded full model checkpoint.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy`.
- LR `3e-3`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 800 steps with snapshots every 100.

Readout cell:

```text
source_seed6_final_additive_seed6_continued_freeze_policy_backbone_steps600
```

Configuration:

- Started from the continued checkpoint above.
- Loaded full model checkpoint.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Ran 600 steps with snapshots every 100.

## Results

| Run | Start final | Continued final | Readout final | Best snapshot | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh `src6` final-source lineage | `0.8975` | `0.9625` | `0.9850` | `0.9900` at readout step `500` | `0.0625` | `0.0547` | `0.9297` | `0.8594` |

Continuation snapshots:

- Step `0`: normal `0.9000`, injection-zero `0.0450`, forced-random `0.0775`.
- Step `500`: normal `0.9625`, injection-zero `0.0450`, forced-random `0.0500`.
- Step `800`: normal `0.9625`, injection-zero `0.0325`, forced-random `0.0475`.

Readout snapshots:

- Step `0`: normal `0.9650`, injection-zero `0.0275`, forced-random `0.0425`.
- Step `500`: normal `0.9900`, injection-zero `0.0575`, forced-random `0.0450`.
- Step `600`: normal `0.9700`, injection-zero `0.0425`, forced-random `0.0350`.

## Conclusion

Label:

```text
src6_selected_continuation_readout_positive
```

The reduced selected-source recipe generalizes to the fresh `src6` source
family. The 600-step-selected final-source handoff was just below gate
(`0.8975`), the extra 800 frozen-policy continuation crossed the gate
(`0.9625`), and the 600-step policy-backbone-frozen readout improved final eval
to `0.9850`.

The result remains calculator-dependent under controls: final injection-zero
was `0.0625` and forced-random was `0.0547`, far below normal `0.9850`. The
calculator-result accuracy stayed near the learned source policy level
(`0.8594`), so the downstream/readout path is using the learned result policy
rather than requiring exact true-sum actions on every example.

## Anti-Rerun Note

Do not repeat this same fresh `src6` final-source continuation plus 600-step
readout recipe as novelty.

Next useful tests:

- validate another fresh source with the 600-step selector and reduced recipe;
- reduce the extra continuation only after fresh-source replications exist;
- optimize source acquisition for 600-step handoff and continuation slope,
  because 500-step selection already failed on this same fresh source.

## Verification

No code changed. The continuation and 600-step readout runs completed and wrote
metrics under the run root above.
