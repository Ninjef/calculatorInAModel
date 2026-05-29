# Phase 7 Seventieth Task: Periodic Review of Selected-Source Recipe

## Status

Completed 2026-05-29.

## Question

Have recent agents been re-running old Phase 7 experiments, or has the work
progressively narrowed the calculator-in-non-bottleneck training recipe?

## Review

Recent work progressed through source-selector failure, handoff-probe
selection, selected-source adaptation, continuation fairness, reduced readout
budget, shorter selector validation, reduced-continuation rejection, and
third-family `src2` selector validation.

## Decision

```text
periodic_review_selected_source_recipe_progressive
```

The recent chain is progressive. It should be treated as a current checkpoint,
not restarted from source normal/calculator accuracy or from already-tested
budget probes.

## Current Recipe

- Use a 500-step frozen-policy handoff probe for checkpoint selection on the
  current source families.
- Keep 800 frozen-policy continuation for weak selected sources.
- Use 600-step no-anchor stable readout adaptation with
  `--freeze-calculator-policy-backbone`.

## Do Not Repeat

- Source normal/calculator accuracy as a primary checkpoint selector.
- Existing `src4/src5` 200/600 readout budget checks as novelty.
- Existing `src4/src5` 600-step continuation cut as novelty.
- Existing `src2` step-1300 versus final 500-step selector audit as novelty.

## Next

Validate the 500-step selector on newly acquired source checkpoints, or train
source acquisition directly for early handoff and continuation slope.
