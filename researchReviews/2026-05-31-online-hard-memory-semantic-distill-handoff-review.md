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

## What Should Stop

- Do not tune semantic-distill weight, sample count, or source length on this
  same op19 seed as novelty.
- Do not run another source-only online-hard-memory job as evidence of
  progress unless it changes the memory/generalization setting.
- Do not return to plain additive semantic distillation without a policy-uptake
  mechanism; that branch already failed.

## What Deserves Compute

- Fresh-seed replication of online-hard-memory plus semantic distillation.
- Streaming/fresh-prompt memory: verify the method is not just fixed-grid
  prompt memorization.
- Larger-range stress after the fresh-seed check.
- Routed/many-calculator validation if the method survives fresh prompts,
  because the final thesis requires scalable many-calculator deployment.

## Strategic Update

This is a real method-combination positive, not a minor variant. Semantic
distillation alone repaired target/readout quality but did not train the
policy; online hard memory alone trained the policy but missed handoff. Their
combination repaired both on the first gate.

The result is still not the full thesis. It uses sparse forced-result scoring
until memory fills and stores per-prompt targets on a fixed grid. The next
research direction should therefore move away from local tuning and toward
generalization/scaling validation.
