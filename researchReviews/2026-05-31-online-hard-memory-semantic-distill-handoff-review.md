# 2026-05-31 - Online Hard Memory Semantic-Distill Handoff Review

## Why This Review

The previous online-hard-memory result was strategically mixed: sparse
zero-improvement memory produced a strong bottleneck source but missed trusted
non-bottleneck handoff. The user also flagged the risk of getting trapped in
minor variations, so this review asks whether adding additive semantic
distillation changed the method-level picture or merely tuned a local knob.

## What Changed

The combined method pairs two previously separate mechanisms:

- sparse answer-derived online hard memory for source-policy uptake;
- arbitrary-result additive semantic distillation for non-bottleneck readout
  geometry.

On the same op19 fixed-grid gate, this combination reached a perfect source
and a perfect trusted frozen-policy additive handoff:

- source final `1.0000`, source calc `1.0000`;
- handoff final `1.0000`, step-600 normal `1.0000`;
- handoff controls low: `0.0525` injection-zero, `0.0050` forced-zero,
  `0.0175` forced-random;
- frozen calculator-result accuracy stayed `1.0000`.

This resolves the immediate failure mode of the prior online-hard-memory run,
where source calc was high but handoff final was only `0.4650`.

Fresh-seed replication changes the strength of that conclusion. On CLI seed
`7` / effective seed `9`, the source again reached `1.0000` final/calc with
low controls and memory frozen after `76,800` forced evals. The trusted
600-step handoff preserved calculator accuracy `1.0000` and low controls, but
missed the pass (`0.6475` final / `0.6625` step-600 normal). A 600-step
continuation improved to `0.8225` final / `0.8500` normal, so the source is
usable but not robustly handoff-friendly.

An alternate downstream handoff seed from the same fresh source also missed
(`0.6325` final / step-600 normal, calculator accuracy `1.0000`, low controls).
This points away from downstream seed luck and toward source/readout geometry.
The cross-source control confirmed this: the original good source paired with
the failed fresh handoff seed reached `1.0000` final / step-600 normal.

A routed shared-output stress adds an important scaling result. The same
combined mechanism, run with four `left_operand_mod` routed hooks and
`--share-calculator-output-proj`, reached source final/calc `1.0000`, trained
all four hooks to calculator-result accuracy `1.0000`, and cleared the trusted
600-step frozen-policy additive handoff at `1.0000` final / step-600 normal.
Controls stayed low: step-600 injection-zero `0.0325`, forced-zero `0.0050`,
forced-random `0.0175`; final 128-sample controls were `0.0391`, `0.0000`,
and `0.0391`. This is the first shared-output routed handoff pass and directly
improves on the prior hard-assignment shared-output handoff misses.

## What Should Stop

- Do not tune semantic-distill weight, sample count, or source length on this
  same op19 seed as novelty.
- Do not repeat the same four-hook shared-output routed seed as novelty.
- Do not run another source-only online-hard-memory job as evidence of
  progress unless it changes the memory/generalization setting.
- Do not return to plain additive semantic distillation without a policy-uptake
  mechanism; that branch already failed.
- Do not describe the combined method as a solved trusted-handoff recipe until
  seed-sensitive handoff/readout behavior is fixed or bounded.

## What Deserves Compute

- Handoff robustness diagnostics: geometry/readout objectives or selection
  criteria that predict the 600-step gate.
- Streaming/fresh-prompt memory: verify the method is not just fixed-grid
  prompt memorization.
- Larger-range stress after the fresh-seed check.
- Fresh routed/shared-output replication or larger-range routed/shared stress,
  because the first routed result used the handoff-friendly seed lineage.

## Strategic Update

This is a real method-combination positive, not a minor variant. Semantic
distillation alone repaired target/readout quality but did not train the
policy; online hard memory alone trained the policy but missed handoff. Their
combination repaired both on the first seed and replicated source acquisition
on a fresh seed.

The result is still not the full thesis. It uses sparse forced-result scoring
until memory fills, stores per-prompt targets on a fixed grid, and still shows
source-geometry-sensitive trusted handoff across seeds. The next research
direction should therefore move away from local tuning and toward fresh
routed/shared validation, streaming/fresh-prompt memory, and larger-range
stress.
