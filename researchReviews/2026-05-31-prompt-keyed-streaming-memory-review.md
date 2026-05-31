# 2026-05-31 - Prompt-Keyed Streaming Memory Review

## Why This Review

Fixed-grid routed/shared online hard memory plus additive semantic distillation
had already cleared op19, fresh op19, and op29. The risk was spending compute
re-proving fixed-grid behavior instead of testing whether the method survives a
more scalable training loop.

## What Changed

The training script now supports prompt-keyed online hard memory with
stochastic training minibatches. This separates three things that had been
coupled:

- exhaustive grid evaluation and expected prompt-count accounting;
- sparse forced-result discovery into hard memory;
- ordinary minibatch optimization of the result policy.

The first 800-step batch64 source filled/froze all `400` memory entries with
true targets, but reached only `0.6325` final and diagnostic calc `0.5781`.
That could have been read as streaming failure. The exposure-matched batch64
source, run for `5000` steps (`5000 * 64` examples, comparable to `800 * 400`
fixed-grid presentations), reached source final/calc `1.0000` and filled/froze
memory after `173,568` forced-result evals. Its trusted frozen-policy additive
handoff reached `1.0000` final / step-600 normal with causal controls.

## What Should Stop

- Do not rerun fixed-grid routed/shared op19 or op29 as novelty.
- Do not rerun the 800-step streaming source as if it disproves the mechanism.
- Do not rerun the same 5000-step op19 streaming source/handoff as novelty.
- Do not treat matched-exposure streaming success as fresh-prompt
  generalization; it still stores per-prompt hard targets.

## What Deserves Compute

1. Fresh/heldout prompt generalization for prompt-keyed memory.
2. Cheaper streaming uptake: fewer optimizer updates after memory fill, replay
   scheduling, or a target/objective that learns as fast as fixed-grid full
   batch without enumerating every prompt every step.
3. A many-calculator scaling gate only if it changes the prompt-memory or
   update-cost story, not just another routed source/handoff repeat.

## Strategic Update

The current method is stronger than the fixed-grid result suggested: it can
train from stochastic minibatches and transfer into a non-bottleneck model. The
main remaining scalability weakness is no longer "must train on the full grid
every step"; it is per-prompt hard memory plus the optimizer-update cost needed
to absorb that memory under minibatching.
