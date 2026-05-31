# Cached Teacher Target Steering Review

Date: 2026-05-30

## What Changed

Frozen-teacher anchoring left two live explanations: online rescoring might
have been the problem, or the policy might need a sharper/easier target. The
cached-table diagnostic separated them.

Cached soft zero-improvement weights reproduced the online anchor. Cached
hard-best teacher targets were much easier for the policy to imitate, reaching
`0.7100` learned-best after 1600 steps. But final eval reached only `0.3725`
because the teacher-best table itself is true for only `0.5225` of prompts.

## What Should Stop

- Same-teacher cached target length/LR/freezing sweeps.
- Treating cached tables as the scalable method; they are a diagnostic unless
  paired with a target construction that can be generated cheaply and correctly.
- More attempts to rescue this semantic-distilled additive teacher table
  without improving its target quality.

## What Deserves Compute

- A better answer-derived target source whose teacher-best table is much closer
  to true arithmetic before policy imitation.
- Cached-table diagnostics for new target sources, because they cheaply reveal
  whether failure is imitation-limited or target-quality-limited.
- Mechanisms that preserve zero-improvement's better target quality while
  making the target easier to imitate than the soft distribution.

## Relation To The Goal

The project has moved one layer deeper: policy uptake is not uniformly broken.
Given an easier hard teacher table, the source policy can imitate it
substantially. The remaining blocker for this branch is target quality and
scalable construction of that target, not repeated scoring overhead.
