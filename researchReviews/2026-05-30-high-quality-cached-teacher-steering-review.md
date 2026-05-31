# 2026-05-30 High-Quality Cached Teacher Steering Review

## Trigger

The additive-path target branch had accumulated several adjacent diagnostics:
semantic readout distillation, frozen-teacher anchoring, cached soft target
weights, cached hard-best, and finally a higher-quality cached teacher table.
This was enough to decide whether more cached-teacher variants should remain a
mainline research direction.

## Current Evidence

- Naive additive-path zero-improvement learned a non-arithmetic target
  (`best_true=0.0325`) and did not train calculator use.
- Semantic readout distillation repaired additive target quality but did not
  produce strong source-policy uptake.
- Frozen-teacher target anchoring preserved a moderate teacher table
  (`best_true=0.5225`) but source final accuracy stayed weak (`0.1750`).
- Cached soft weights reproduced the weak uptake problem and showed repeated
  rescoring was not the main bottleneck.
- Cached hard-best was easier to imitate, and a higher-quality teacher table
  improved source final accuracy to `0.5825`, but still remained below the
  teacher's `0.8200` best-true ceiling.

## Steering Decision

Cached teacher tables should be treated as diagnostics, not as the recipe. The
branch has now shown two important things:

1. Target-table quality is a real limiter.
2. Hard target imitation is much easier than soft target-weight imitation.

But cached hard-best still depends on full-enum teacher scoring, does not
produce scalable credit assignment, and does not close the gap to the teacher
table. More same-teacher length, LR, temperature, or cache-mode sweeps would be
local polish rather than progress toward the project goal.

## Future Work

Spend compute on mechanisms that create high-quality answer-derived targets
without a fixed full-enum teacher cache. Acceptable continuations include:

- A genuinely new online/scalable target-construction mechanism.
- A proposal/training co-design that preserves hard-target quality at much
  lower candidate cost.
- A handoff-aware bottleneck source objective whose target quality can be
  validated with cached hard-best only as a quick diagnostic.

Avoid:

- Same-teacher cached hard-best length/LR sweeps.
- Retrying soft cached target weights without a materially different loss.
- Treating cached/full-enum teacher tables as a scalable training solution.
