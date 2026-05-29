# 2026-05-29 New-Source 500-Step Selector Validation

## Aim

Validate the shortened 500-step handoff selector on a newly acquired source
family, rather than only on existing `src2/src4/src5` traces.

## Source Acquisition

Run root:

```text
runs/2026-05-29_phase7_new_source_500_selector_validation
```

Source cell:

```text
source_seed6_snapshots_steps1600
```

Configuration matched the recent bottleneck source recipe:

- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=direct_feedback_alignment`
- `calculator_action_head=result_space`
- frozen product semantic decoder
- `result_policy_improvement_assignment_weight=10`
- exact-grid natural `0..19`
- CLI seed `6`
- 1600 steps with checkpoint snapshots every 100 steps

An initial launch without the tiny architecture failed before training because
the semantic decoder checkpoint is `n_embd=16`; the successful run used
`n_layer=2`, `n_head=1`, `n_embd=16`, `mlp_expansion=1`, and hook layer `1`.

Source diagnostics:

| Source checkpoint | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| step `1200` | `0.8350` | `0.0500` | `1.0000` | `0.8350` |
| step `1500` | `0.8625` | `0.0450` | `1.0000` | `0.8625` |
| step `1600` snapshot | `0.8650` | `0.0325` | `1.0000` | `0.8650` |
| final eval | `0.8850` | `0.0391` | `1.0000` | `0.8594` |

## Additive Handoff Candidates

All handoffs used additive non-bottleneck mode, compatible checkpoint load,
`--freeze-calculator-policy`, exact-grid natural `0..19`, CLI seed `6`, and
800 training steps. The 800-step runs give both the 500-step selector score and
the full handoff confirmation.

| Candidate | Source normal | Normal @ 400 | Normal @ 500 | Normal @ 600 | Normal @ 800 | Final eval | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `1200` | `0.8350` | `0.4025` | `0.6025` | `0.7075` | `0.8025` | `0.8450` | `0.0000` | `0.0703` | `0.8438` | `0.7422` |
| step `1500` | `0.8625` | `0.5450` | `0.7200` | `0.7800` | `0.8500` | `0.8875` | `0.0000` | `0.0469` | `0.8438` | `0.8359` |
| final | `0.8850` | `0.5950` | `0.6850` | `0.8050` | `0.8975` | `0.8975` | `0.0469` | `0.0703` | `0.9063` | `0.8594` |

## Conclusion

Label:

```text
new_source_500_step_selector_generalization_negative
```

The 500-step selector does not generalize cleanly to this newly acquired
source. It would select step `1500` (`0.7200` at 500 steps), but the full
800-step handoff was better from the final source checkpoint (`0.8975` versus
`0.8875` final eval).

The 600-step selector would have picked the final checkpoint (`0.8050` versus
`0.7800`), matching the full-handoff winner. This updates the recipe boundary:
500-step selection remains a useful trace shortcut on `src2/src4/src5`, but
new source families should keep the 600-step selector or require full
confirmation until a cheaper proxy is validated.

The fresh final-source handoff was near the `0.90` gate and retained a
calculator-use signature relative to controls, but did not cleanly pass the
gate (`0.8975` final eval, injection-zero `0.0469`, forced-random `0.0703`).

## Anti-Rerun Note

Do not repeat this same `src6` step `1200/1500/final` additive seed-6 frozen
handoff comparison as novelty.

Next useful tests:

- use 600-step selection, not 500-step selection, for fresh source families;
- run the selected-source continuation/readout recipe from the fresh `src6`
  final handoff to test whether the near-gate source clears `0.90`;
- optimize source acquisition for 600-step handoff/continuation slope instead
  of source normal accuracy or the now-overfit 500-step shortcut.

## Verification

No code changed. The source acquisition and all three 800-step handoff
candidate runs completed and wrote metrics under the run root above.
