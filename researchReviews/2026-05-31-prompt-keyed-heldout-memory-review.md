# 2026-05-31 - Prompt-Keyed Heldout Memory Review

## Why This Review

The previous streaming review showed that prompt-keyed online hard memory can
train routed shared-output calculators from stochastic minibatches when update
exposure is matched. That left the strategic question the user raised: are we
still doing high-leverage method tests, or just repeating one successful family?

## What Changed

A deterministic heldout-prompt gate now separates transductive memory from
fresh-prompt generalization. The same four-hook shared-output streaming recipe
trained only on `320` of `400` op19 prompts for 5000 batch64 steps. It filled
and froze exactly those `320` prompt-memory entries after `87,552` forced-result
evals.

The train prompts nearly solved: exact/calc `0.996875`, `319/320` correct. The
`80` heldout prompts, absent from both minibatches and prompt memory, reached
only `0.0875` exact/calc, `7/80` correct, with low controls.

## Method Count Check

Strictly, the project has tried about thirteen actual training-method families.
The May 30 method review counted ten. Since then, three method-level additions
deserve to be counted: answer-derived zero-improvement/result-boundary targets,
additive semantic distillation as a readout-geometry auxiliary, and sparse
online hard result-boundary memory. Frozen/cached teacher tables are better
classified as diagnostics for target quality and policy uptake, not a separate
mainline method.

The recent routed/shared, op29, streaming matched-exposure, and heldout runs are
not separate methods. They are validation and boundary tests of the online hard
memory plus semantic distillation family.

## What Should Stop

- Do not run a trusted handoff from this heldout-failed source as progress.
- Do not treat prompt-keyed memory as a fresh-prompt generalization mechanism.
- Do not rerun same-exposure op19 streaming memory unless the run adds a
  genuinely non-transductive credit mechanism.
- Do not spend another cycle on fixed-grid routed/shared op19/op29 validation;
  that family is already proven on seen prompts.

## What Deserves Compute

1. Amortized target discovery: train a proposal, initializer, or critic that can
   supply calculator targets for prompts before they have per-prompt memory.
2. Fresh-prompt candidate scoring: allow sparse answer-derived discovery on new
   prompts while measuring cost and heldout uptake explicitly.
3. A learned memory initializer or state-conditioned target prior that is gated
   by deterministic heldout prompts before any handoff run.

## Strategic Update

We have a strong seen-prompt calculator-training method: online hard
result-boundary memory plus additive semantic distillation, including routed
shared-output calculators, matched-exposure streaming, and non-bottleneck
handoff. The active blocker is now fresh-prompt credit assignment. Future work
should be counted as a new training method only if it changes how targets are
created or generalized beyond stored prompt keys.
