# 2026-05-29 Periodic Review: Selected-Source Recipe

## Aim

Check whether the recent Phase 7 work is re-running old ideas or making
progress, and pin down the current non-bottleneck recipe so the next work does
not backslide.

## Review Window

Reviewed the recent source-selection and selected-source adaptation chain:

- Source normal/calculator accuracy selector failure on `src2`, `src4`, and
  `src5`.
- Short handoff-probe audits and selector validations.
- Probe-selected stable-policy adaptation.
- Continuation fairness checks for `src4` and `src5`.
- Readout, handoff-probe, and continuation budget reductions.

## Anti-Rerun Finding

The recent work is progressive, not regressive. It moved through this chain:

1. Source accuracy was tested and disproven as a reliable selector.
2. Short downstream handoff progress became the working selector.
3. Probe-selected checkpoints were validated on weak `src4/src5` handoffs.
4. Continuation-depth fairness explained the old `src5` final-source advantage.
5. The stable readout budget was cut from 1600 to 600 steps.
6. The selector was cut from 600 to 500 steps on current traces.
7. A 600-step continuation cut was rejected for weak `src4`.
8. The 500-step selector was validated on the separate `src2` counterexample.

This is a useful narrowing sequence. The work should not return to source
normal/calculator accuracy as the primary source-checkpoint selector.

## Current Validated Non-Bottleneck Recipe

For the currently tested selected-source lineages:

1. Train bottleneck source with checkpoint snapshots.
2. Select source checkpoint by a 500-step frozen-policy additive handoff probe.
3. Run the selected 800-step frozen-policy handoff.
4. For weak selected sources, keep an extra 800 frozen-policy continuation.
5. Run 600-step no-anchor readout adaptation with
   `--freeze-calculator-policy-backbone`.

Known outcomes:

| Source lineage | Selected source | Continuation | Readout | Final eval | Controls |
| --- | --- | ---: | ---: | ---: | --- |
| `src4/add2` | step `1200` | 800 | 600 | `0.9025` | injection-zero `0.0025`, forced-random `0.0175` |
| `src5/add5` | step `1100` | 800 | 600 | `0.9325` | injection-zero `0.0000`, forced-random `0.0250` |
| `src2/add4` | final / step `1600` | not needed in this audit | 800 frozen handoff only | `0.9525` | source-accuracy counterexample validates selector |

## Remaining Gaps

- This is still post-hoc source selection, not end-to-end answer-loss discovery.
- The bottleneck source acquisition stage remains prescriptive/hard-assignment
  relative to the final nice-to-have.
- The 500-step selector is validated on current traces, but not yet on newly
  acquired source checkpoints.
- The extra 800 frozen-policy continuation is still needed for weak `src4`;
  the 600-step continuation cut failed the final-eval gate.
- Scaling to many calculators and larger models remains unproven.

## Next Non-Duplicative Direction

Train or acquire a new source with snapshots, select by the 500-step handoff
probe, then run the reduced selected-source recipe once. Prefer adding a source
acquisition objective or selection loop that directly rewards early handoff and
continuation slope. Avoid repeating the existing `src4/src5` 200/600 readout,
600 continuation, or source-accuracy selector comparisons as novelty.

Label:

```text
periodic_review_selected_source_recipe_progressive
```
