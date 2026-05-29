# 2026-05-29 Phase 7 Strategy Review

## What Changed

Phase 7 has moved from architecture validation to credit-assignment research.

The project has enough evidence that the calculator path can work:

- downstream answer layers can use correct calculator outputs;
- identifiable or scaffolded calculator protocols can be learned and retained;
- hard improvement assignment can train natural result-level bottleneck
  calculator requests;
- frozen/protected bottleneck policies can be handed into additive
  non-bottleneck models and used causally.

The accumulating negatives are now more important than the local positives.
Ordinary answer-loss discovery and many nearby proxy/selector variants have
not produced a scalable route to calculator-query discovery.

## Belief Changes

The central bottleneck is not model capacity, calculator wiring, or downstream
decoding. It is the learning signal for upstream calculator queries.

The best current non-bottleneck recipe is real but not final:

```text
bottleneck hard-assignment source
-> source checkpoint selection by actual handoff behavior
-> additive transfer with frozen/protected policy
-> continuation/readout
```

This proves viability of staged calculator use, but it remains costly,
checkpoint-sensitive, and partly prescriptive.

## Branches That Should Stop

Do not keep spending mainline effort on these without a new mechanism:

- oracle-only success;
- generic retention-after-teaching;
- vanilla REINFORCE or expected answer-loss variants;
- decoder calibration alone;
- simple direct feedback and simple shadow-gradient tweaks;
- cheap checkpoint selectors that do not beat actual handoff behavior;
- source accuracy as a transfer selector;
- small source-weight or schedule tweaks without a transfer-geometry target.

## Branches That Deserve Compute

Two classes deserve attention:

1. Source acquisition that directly optimizes or regularizes for downstream
   additive handoff/readout geometry.
2. A genuinely different credit-assignment family, such as target propagation
   or local targets, that can construct useful calculator-query targets without
   differentiating through the calculator.

A lower-cost hard-assignment approximation is also worth considering only if
it directly addresses scalability and is compared to the exact-grid assignment
ceiling.

## Are We Closer To The Goal?

Yes, but not because the final method is known.

We are closer because the failure surface is clearer:

- the architecture can host useful calculator policies;
- non-bottleneck causal calculator use is achievable by staged transfer;
- the missing piece is scalable, non-prescriptive credit assignment;
- many local variants around ordinary answer loss are no longer promising.

The next progress should be strategic, not merely incremental.

## What Counts As Success Next

A useful next result would do at least one of these:

- show early from-scratch calculator-query learning from an answer-derived
  signal;
- improve fresh-source handoff/readout behavior by optimizing transfer geometry
  during source acquisition;
- approximate hard improvement assignment at lower cost without losing the
  bottleneck source result;
- demonstrate a target-propagation/local-target mechanism with a meaningful
  Stage 0 gate and early Stage 1 lift.

## Steering Decision

Mainline work should stop treating proxy-selector refinement as the default
frontier. The project should re-steer toward credit assignment or
transfer-geometry source acquisition.
