# 2026-05-29 Forced-Margin Branch Review

## Why This Review Exists

The additive forced-margin branch has now gone through enough gates to stop
treating each next knob as fresh strategy:

1. Small `operand_max=9` mechanism gate.
2. Full-grid `operand_max=19` matched 200-step source gate.
3. Full-grid compute check for 4 negatives.
4. One-negative trusted 600-step handoff.
5. Longer one-negative source horizon.
6. Continuation from the exact positive 200-step checkpoint.

This review decides what the branch has learned, what should stop, and what
deserves more compute.

## What Changed

The forced-margin objective is a real source-geometry mechanism.

- Small gate: scheduled 4-negative margin improved forced-result ranking
  (`forced_best_true=0.6200`, top3 `0.7500`) without hurting source policy
  acquisition, but had worse 50-step slope than scheduled forced-true.
- Full-grid 4-negative attempt: too costly locally; it wrote only config after
  roughly ten minutes, so many-negative full-grid margin is not the scalable
  branch unless the implementation or schedule changes.
- Full-grid one-negative gate: practical and positive. At the matched
  200-step source point it reached `0.3225` source calc / `0.3600` source final
  eval and `0.6600` trusted 600-step handoff final eval, beating matched
  scheduled forced-true (`0.4150`) and baseline (`0.2525`).
- Longer one-negative source: improved the branch but did not clearly beat
  scheduled forced-true step-600. Fresh step-600 source handoff reached
  `0.7330` final / `0.7500` step-600 normal. Continuing the exact positive
  step-200 checkpoint found a better intermediate handoff (`0.7400` final /
  `0.7850` step-600 normal) but another 200 source steps degraded source final
  eval back to `0.3600`.

## What Should Stop

- Do not rerun full-grid 4-negative forced-margin without a compute-reduction
  plan. It is locally too expensive and violates the scalability pressure that
  motivated the branch.
- Do not rerun seed-13 one-negative longer-horizon ladders, continuation from
  the same step-200 checkpoint, or handoffs from the tested step-400/step-600
  / continued-step-200 checkpoints as novelty.
- Do not tune negative count, start step, slope probe length, or source
  checkpoint selection as a local forced-margin variant unless the experiment
  changes the source-policy bottleneck or validates on a fresh trusted
  600-step handoff outcome.
- Do not treat forced-result geometry or 50-step slope as selectors. They were
  useful diagnostics, but the branch again showed that actual 600-step handoff
  is decisive.

## What Deserves Compute

Compute is still justified if the question is one of these:

- Source-policy bottleneck: add a predeclared late recovery/retention phase to
  the one-negative margin branch, then verify with the trusted 600-step handoff
  and, only if it passes, continuation/readout.
- Stability: replicate the one-negative margin source on a fresh seed with the
  same trusted 600-step handoff gate. This is useful only if the question is
  robustness, not another same-seed improvement.
- Scalability/non-prescriptiveness: use the branch as evidence that transfer
  geometry matters, but move mainline effort toward learned/generalized local
  proposals, estimator correction, or a target construction that does not
  require hand-picked true-result forcing.

## Are We Closer?

Yes, but only in the staged-transfer sense.

The branch found a cheaper source-geometry auxiliary that improves early
full-grid handoff and keeps controls low. That is progress toward reliable
non-bottleneck calculator use. It does not solve the overarching goal because
the method still depends on hard improvement assignment and a prescriptive
true-result forced auxiliary, and the best branch remains checkpoint-sensitive.

## Steering Decision

Treat one-negative forced-margin as a useful constrained auxiliary, not the new
mainline by itself.

If the next experiment stays in this branch, it must explicitly test source
recovery/retention or fresh-seed stability. Otherwise, move effort back toward
less prescriptive, scalable credit assignment while preserving the rule that
actual 600-step handoff/readout gates arbitrate source-geometry claims.
