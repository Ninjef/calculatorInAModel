# 2026-05-29 `src7` 600-Step Selector Replication

## Aim

Replicate the fresh-source path after the `src6` result: use 600-step handoff
selection on a newly acquired source, then test whether the reduced
continuation/readout recipe clears the non-bottleneck gate.

## Source Acquisition

Run root:

```text
runs/2026-05-29_phase7_src7_600_selector_replication
```

Source cell:

```text
source_seed7_snapshots_steps1600
```

Configuration matched the current bottleneck source recipe:

- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=direct_feedback_alignment`
- `calculator_action_head=result_space`
- frozen product semantic decoder
- `result_policy_improvement_assignment_weight=10`
- exact-grid natural `0..19`
- CLI seed `7`
- 1600 steps with checkpoint snapshots every 100 steps

Source diagnostics:

| Source checkpoint | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| step `1000` | `0.7075` | `0.0500` | `1.0000` | `0.7075` |
| step `1400` | `0.7500` | `0.0325` | `1.0000` | `0.7500` |
| step `1600` snapshot | `0.8050` | `0.0300` | `1.0000` | `0.8050` |
| final eval | `0.8100` | `0.0703` | `1.0000` | `0.7344` |

## Additive Handoff Candidates

All handoffs used additive non-bottleneck mode, compatible checkpoint load,
`--freeze-calculator-policy`, exact-grid natural `0..19`, CLI seed `7`, and
800 training steps.

| Candidate | Source normal | Normal @ 600 | Normal @ 800 | Final eval | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `1000` | `0.7075` | `0.4850` | `0.5450` | `0.5375` | `0.0000` | `0.0234` | `0.6563` | `0.7266` |
| step `1400` | `0.7500` | `0.5025` | `0.6600` | `0.7325` | `0.0234` | `0.0391` | `0.7188` | `0.7891` |
| final | `0.8100` | `0.4150` | `0.5200` | `0.5000` | `0.0000` | `0.0078` | `0.5313` | `0.7344` |

The 600-step selector chooses step `1400`, and step `1400` also wins the full
800-step handoff. Source normal accuracy alone would have preferred the final
checkpoint, which transferred worse.

## Selected Continuation and Readout

Selected lineage:

```text
source_seed7_step1400_additive_seed7
```

| Stage | Final eval | Best snapshot | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial handoff | `0.7325` | `0.6600` at step `800` | `0.0234` | `0.0391` | `0.7188` | `0.7891` |
| 800-step continuation | `0.8125` | `0.8050` at step `700` | `0.0469` | `0.0391` | `0.7734` | `0.7891` |
| 600-step readout | `0.8825` | `0.8650` at step `400` | `0.0625` | `0.0703` | `0.8125` | `0.7891` |

## Conclusion

Label:

```text
src7_600_step_selector_positive_recipe_boundary_negative
```

The 600-step selector replicated on a harder fresh source: it chose step
`1400`, matching the full-handoff winner and avoiding the higher-source-normal
final checkpoint.

The reduced continuation/readout recipe did not clear the `0.90`
non-bottleneck gate on this source. It improved the selected lineage from
`0.7325` to `0.8125` after continuation and `0.8825` after readout, with
controls still far below normal, but the result remains below gate.

This is evidence that source acquisition quality is now a limiting factor:
the selector can pick the best available checkpoint, but a weak source family
does not reliably produce enough downstream handoff/readout headroom.

## Anti-Rerun Note

Do not repeat this same `src7` step `1000/1400/final` additive seed-7 handoff
comparison or the same step-1400 continuation/readout chain as novelty.

Next useful tests:

- optimize source acquisition directly for 600-step handoff/continuation slope;
- acquire additional fresh sources only if they change the acquisition recipe
  or provide a planned replication gate;
- avoid returning to source normal/calculator accuracy as a selector.

## Verification

No code changed. Source acquisition, three handoff candidates, selected
continuation, and selected readout all completed and wrote metrics under the
run root above.
