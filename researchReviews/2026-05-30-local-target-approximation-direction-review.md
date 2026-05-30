# 2026-05-30 Local-Target Approximation Direction Review

## Why This Review Exists

The local-target branch is at risk of re-entering the same loop under new
names: sparse candidates, proposal tweaks, replay tables, learned proposals,
and now sparse pairwise preferences.

This review consolidates the family-level evidence after the sampled
pairwise-preference negative so future work treats exact local targets as a
ceiling/diagnostic, not the current scalable path.

## What Changed

The positive remains important:

- Exact `policy_reweighted_t1` trains natural result-level calculator use and
  survives answer-only retention.
- It is the cleanest proof that answer-derived local targets can push the
  calculator policy where ordinary expected answer loss fails.

The approximation story has weakened:

- Naive sparse uniform/top-k candidates require near-full result coverage.
- Low-loss neighborhood expansion clustered candidates and underperformed raw
  uniform `u32`.
- Fixed replay memory produced strong fixed-grid results, but reset and
  streaming stress exposed prompt-transductive dependence.
- Imputed unscored-mass corrections diluted target pressure.
- A simple online learned proposal won on the fixed grid, then tied raw `u32`
  under 800-step streaming minibatches.
- Random-prompt proposal pretraining nudged exact calc but hurt sampled normal.
- Sparse pairwise preferences changed the target construction, yet still
  failed the 200-step Stage 1 gate: `u32` reached only `0.0425` exact calc
  despite `0.8450` true-candidate coverage, while same-budget
  policy-reweighted `u32` reached `0.3350`.

## Belief Changes

The branch has answered the "can simple sparse approximations recover exact
local-target learning?" question strongly enough for now: no.

The failure is not just a candidate-count issue. Pairwise preference saw the
true result in most prompts at `u32` and still produced almost no useful
calculator policy. Fixed-grid learned/replay positives are useful diagnostics,
but they do not establish scalable credit assignment for many calculators,
larger models, or streaming prompt distributions.

Exact `policy_reweighted_t1` should remain as a ceiling, comparison target, and
source of mechanism clues. It should not remain the active compute mainline
unless the next idea changes the estimator or target construction in a way that
does not depend on near-full candidate inclusion.

## What Should Stop

Do not treat these as novel:

- another sparse candidate-count ladder for `policy_reweighted_t1`;
- top-k, neighborhood, or simple proposal-mixture variants;
- fixed/prompt-keyed replay cache variants, reset windows, or rescore counts;
- mean/current/max imputed unscored-mass corrections;
- the same polynomial-feature learned proposal with more hidden units, epochs,
  warmup counts, or seed replications;
- sparse pairwise-preference count or loss-gap sweeps;
- fixed-grid-only positives without predeclared streaming/generalization
  validation.

## What Deserves Compute

Local-target work deserves renewed compute only for a mechanism-level change:

- an estimator with a stated bias/variance story and a gate against raw `u32`
  plus exact `policy_reweighted_t1`;
- a target that creates useful pressure when the useful result is absent from
  the sampled candidate set;
- uncertainty-aware or active compute allocation that is evaluated against a
  full-grid ceiling and streaming prompts;
- a proposal/generalizer whose validation objective is heldout prompt or
  evolving-model generalization, not current fixed-grid coverage.

Otherwise, near-term compute should pivot to source objectives that improve
actual additive handoff/readout behavior or to less-prescriptive
answer-derived target construction that replaces full forced-result
enumeration.

## Are We Closer To The Goal?

Yes, by closing a seductive loop.

The project now has a sharper separation between a useful proof-of-principle
ceiling and a scalable training method. Exact local targets show the kind of
credit assignment we want, but the tested sparse/proposal/pairwise families do
not make it scalable.

## Steering Decision

Pause local-target approximation as a mainline compute branch.

Keep exact `policy_reweighted_t1` for ceilings and diagnostics. Continue only
with a materially different estimator, target construction, or explicit
streaming/generalization validation. In the absence of that mechanism, spend
the next compute on source-geometry objectives or answer-derived boundary
methods that reduce enumeration without returning to simple proposal tuning.
