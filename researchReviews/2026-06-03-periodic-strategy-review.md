# 2026-06-03 Periodic Strategy Review

## Why This Review Exists

The user asked for periodic zoom-outs to make sure future agents do not keep
rerunning old theories with slightly different knobs. The last few days
produced a strong benchmark family, several cost reductions, and a cluster of
route-excluded variants. This review decides what should stop and what should
count as the next real move toward the goal.

## Scope

Reviewed evidence:

- Training method count and current family-14 status.
- Capped-prior many-calculator accounting.
- Route-heldout shared-prior diagnostic.
- Corrected route-excluded source, route replay, prior bootstrap,
  candidate-evidence, and background evidence refresh.
- Paused local-target/proposal and source-selector reviews.

## Main Pattern

The project has not been stuck because calculators cannot be wired or because
handoff is impossible. The strongest staged recipe now trains bottleneck
calculator policies from scratch and transfers them through trusted
frozen-policy additive handoff. The remaining gap is narrower and harder:
current success depends on answer-derived candidate scoring, per-prompt or
per-calculator target memory, numeric-prior fitting, and staged protection.

Recent work repeatedly tried to polish that family:

- Capped/proportional/full-refresh prior fitting reduced some cost and produced
  a strong op29 benchmark, but many-calculator accounting still scales linearly
  with independent calculators.
- Route-heldout diagnostics showed numeric priors can share target structure
  offline, but live route-excluded source training did not convert that into a
  heldout-generalizing source.
- Route replay, prior bootstrap, candidate-evidence, and background refresh all
  changed local pressure/timing around the same prompt-memory target system.
  None cleared the route-excluded source gate; background refresh made it worse
  despite heavy extra scoring.
- Earlier local-target approximation and replay-cache positives were real on
  fixed grids, but failed or weakened under streaming/generalization stresses.

## Decisions

Close these as next-step branches:

- Route-excluded shared-prior patching: no route-replay weights, bootstrap
  thresholds/caps, candidate-evidence timings, or background-refresh
  batch/every/weight ladders.
- Family-14 cost polishing that only changes cap value, refresh window,
  proportional fraction, validation threshold, hidden size, or same-recipe seed.
- Cheap source selectors and embedded handoff proxies as decision makers.
- Static local-target/proposal variants unless they introduce a materially new
  estimator or target construction with a predeclared streaming/heldout gate.

Do not count those as new algorithms. They are knobs around the current
prescriptive target-memory recipe.

## Next Compute Bar

The next experiment should be rejected unless it changes at least one of these
mechanisms:

- Target formation: e.g. a shared/global target model that learns across routes
  before per-route memories freeze, rather than independent prompt tables.
- Cross-calculator amortization: e.g. one target/prior learner whose update
  count does not scale linearly with calculator count.
- Credit assignment: e.g. a less-prescriptive signal that bypasses
  answer-derived candidate scoring or true-result forcing.
- Transfer objective: e.g. source training directly optimized for the trusted
  non-bottleneck readout/handoff geometry in a way that is not just a
  checkpoint selector.

A run can still use the current family as a benchmark or control, but the
research claim must be about one of the changes above.

## Suggested Next Directions

1. Design a global target/prior learner shared across routes/calculators, then
   test whether disabling direct target discovery on a subset no longer breaks
   heldout source quality.
2. Prototype a credit signal that trains calculator results from answer loss
   without enumerating/scoring a prompt-specific candidate set.
3. If staying staged, make the source objective explicitly target trusted
   additive readout geometry and validate with the full 600-step handoff gate,
   not with source accuracy or cheap selectors.

## Bottom Line

We are closer on architecture, bottleneck source training, and staged
non-bottleneck transfer. We are not closer enough on the actual final goal until
the method stops depending on per-calculator prompt-memory target tables and
answer-derived candidate scoring. Future agents should spend compute only on
that break, not on another local variant of the current recipe.
