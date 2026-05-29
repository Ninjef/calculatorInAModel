# 2026-05-29 Phase 7 Local-Target Approximation Review

## Why This Review Exists

The local-target branch produced a real positive and then immediately risked
becoming a local variant loop.

Exact `policy_reweighted_t1` targets showed Stage 1 lift and answer-only
retention, but the method still scores broad forced-result candidates. The
last cluster tested whether simple sampled/adaptive candidate sets could make
that signal scalable. They did not.

This review freezes the lesson before future agents keep trying nearby
hand-coded candidate proposals.

## What Changed

The local-target branch is no longer merely a feasibility idea.

- Stage 0 showed current-policy-reweighted targets align with the hard
  boundary ceiling while ordinary expected answer loss anti-aligns.
- A 200-step Stage 1 gate showed `policy_reweighted_t1` reaches `0.5600`
  exact-grid calculator-result accuracy and `0.5391` sampled normal.
- An 800-step target-training plus 200-step answer-only retention gate showed
  the branch can recover and finish at `0.8925` exact-grid calculator-result
  accuracy and `0.8750` sampled normal.

But the scalability story got worse:

- Naive sparse no-replacement sampling did not preserve the learning signal
  until candidate coverage approached full enumeration. `u32` scored about
  `82%` of the result vocabulary and still reached only `0.3350` exact-grid
  calc at 200 steps.
- Full-vocabulary sparse-path `u39` recovered the signal, confirming the
  implementation was sound and the failure was coverage/proposal quality.
- Current-policy top-k plus uniform sampling was worse than uniform-only
  sampling at comparable budget.
- Loss-ranked neighborhood expansion also underperformed raw uniform `u32`
  because it clustered into fewer unique candidates and lower true-result
  coverage.

## Belief Changes

The local-target branch remains strategically interesting because it changes
the credit-assignment family. It is the best recent evidence that an
answer-derived local target can train and retain a natural result-level
calculator policy.

However, simple hand-coded candidate proposal is not the path to scalability.
The target seems to need either high true-result candidate coverage or a
different estimator/target construction that does not collapse when the true
useful result is absent from the candidate set.

The main uncertainty is now sharper:

```text
Can we learn or correct the proposal/estimator enough to avoid near-full
forced-result scoring?
```

If not, the local-target branch remains another prescriptive full-enum
assignment method, not the final scalable answer.

## Branches That Should Stop

Stop running hand-coded sampled-candidate ladders as novelty:

- raw uniform/top-k count ladders;
- top-k plus uniform variants without a new correction mechanism;
- low-loss integer-neighborhood expansion;
- longer versions of the same sparse candidate proposals;
- seed replications of these sparse proposals unless a new estimator or
  proposal mechanism changes the expected failure mode.

The exact `policy_reweighted_t1` branch also should not be rerun merely to
reconfirm retention. It is positive but non-scalable.

## Branches That Deserve Compute

Three local-target follow-ups still deserve compute if they are made explicit:

1. A learned candidate proposal with a predeclared coverage and Stage 1 gate.
   It should beat raw uniform `u32` on true-result coverage and calculator
   learning at similar scoring cost.
2. An importance-corrected or bias-corrected sampled target that changes the
   estimator, not just the candidate count.
3. A different local-target construction that does not require the useful
   result to be present in a small candidate set.

If none of these is concrete, the better mainline path is source acquisition
for additive handoff/readout geometry, where the project already has a staged
non-bottleneck recipe but needs better source objectives.

## Are We Closer To The Goal?

Yes, but mostly by ruling out a tempting scalability shortcut.

The exact local-target branch shows that a non-backpropagated answer-derived
target can train calculator use. The sparse and adaptive negatives show that
making this scalable is not as simple as sampling a few candidate results.

The project is closer because future work can avoid re-entering the
candidate-count loop and focus on either learned/corrected proposals or the
source-acquisition-for-handoff branch.

## What Counts As Success Next

A next local-target result should satisfy all of:

- score materially fewer result classes than full enumeration;
- beat raw uniform `u32` and adaptive-neighborhood baselines at comparable
  scoring cost;
- show Stage 1 calculator-result lift, not only target coverage;
- report true-result candidate coverage, target argmax accuracy, sampled
  normal, injection-zero, forced-random, and oracle controls.

A next source-acquisition result should instead improve fresh-family
standalone 600-step handoff or continuation/readout behavior, not merely source
calculator accuracy.

## Steering Decision

Keep target propagation/local targets active, but narrow the allowed work:

- exact-grid `policy_reweighted_t1` is a positive ceiling, not a scalable
  method;
- hand-coded sparse/adaptive proposals are paused;
- local-target approximation needs a learned proposal, estimator correction, or
  different target construction;
- absent that, pivot mainline compute to source acquisition optimized for
  additive handoff/readout geometry.
