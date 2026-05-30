# 2026-05-29 Local-Target Proposal Branch Review

## Why This Review Exists

The local-target branch produced the strongest recent answer-derived
credit-assignment positive, then accumulated enough approximation attempts to
risk becoming a proposal-variant loop.

This review consolidates the fixed-grid, replay-memory, streaming, corrected,
and learned-proposal results so future work does not keep returning to old
coverage/proposal theories under new branch names.

## What Changed

The positive is real:

- Exact `policy_reweighted_t1` trains a natural result-level calculator policy
  and can survive answer-only retention.
- Replay memory and learned proposal variants showed that candidate proposal
  can matter: both produced fixed-grid 200-step results near or above the exact
  local-target ceiling.

The scalability interpretation changed:

- Naive sparse/top-k/adaptive proposals failed unless coverage approached full
  enumeration.
- Fixed replay memory was transductive. Reset windows damaged it, and
  streaming minibatches removed the strong lift.
- Preserving unscored policy mass with imputed losses diluted target pressure
  and underperformed raw uniform `u32`.
- A simple online learned loss proposal beat raw `u32` on the fixed grid
  (`0.5850` exact calc / `0.5703` sampled normal), but did not beat raw `u32`
  under 800-step streaming minibatches.
- Random-prompt proposal pretraining only nudged streaming exact calc
  (`0.2625` vs `0.2350`) while hurting sampled normal (`0.1797` vs `0.2734`).

## Belief Changes

The local-target family remains important as a ceiling and proof of principle:
answer-derived local targets can train calculator use where ordinary expected
answer-loss gradients fail.

But simple candidate proposal is not the missing scalable mechanism. Every
variant that still depends on "find enough useful result candidates, then
apply the same sparse policy-reweighted target" has either failed streaming
stress or depended on prompt-specific transduction.

The branch should stop asking:

```text
Can we pick a better small candidate set with the same target?
```

It should now ask:

```text
Can we change the estimator or target so useful pressure exists without
near-full true-result candidate coverage?
```

## What Should Stop

Do not treat these as novel:

- raw uniform/top-k count ladders;
- low-loss neighborhood expansion;
- fixed or prompt-keyed replay-memory budget, rescore, reset, batch-size, or
  longer-run variants;
- mean/current/max imputed unscored-mass corrections;
- the same polynomial-feature learned proposal MLP, with or without `_wN`
  random-prompt warmup;
- fixed-grid-only learned proposal results without streaming lift;
- seed replications of these proposal mechanisms unless a new estimator or
  target construction is being validated.

## What Deserves Compute

Local-target compute is justified only for a mechanism-level change:

- an estimator with an explicit bias/variance argument and a gate against raw
  `u32` plus exact `policy_reweighted_t1`;
- a target construction that creates useful gradients when the true useful
  result is absent from a small candidate set;
- a learned proposal whose validation objective is about streaming/full-grid
  generalization, not just current-batch coverage.

Otherwise, mainline compute should pivot back to source acquisition for
additive handoff/readout geometry, where the project has a working staged
non-bottleneck recipe and concrete next questions around source objectives,
fresh-seed robustness, and less-prescriptive assignment.

## Are We Closer To The Goal?

Yes, by removing a tempting path.

The project learned that exact local targets can train calculator use, but the
scalable approximation is not solved by better candidate coverage alone. That
keeps the overarching goal honest: scalable training for many calculators or
larger models cannot rely on prompt tables, near-full result scoring, or
fixed-grid proposal wins.

## Steering Decision

Pause simple local-target proposal approximation as a mainline branch.

Keep exact `policy_reweighted_t1` as a ceiling and diagnostic. Continue local
targets only with a different estimator, a different target construction, or a
learned proposal whose validation is explicitly streaming/generalization
oriented. In the absence of that concrete mechanism, spend the next compute on
source objectives that improve actual 600-step additive handoff/readout
behavior.
