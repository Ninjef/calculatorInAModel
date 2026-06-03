# 2026-06-03 Current-Batch Prior Target Review

## Why This Review Exists

The project just added current-batch prior-target tooling after a periodic
review closed local proxy polishing and route-excluded patch variants. This
review decides whether that tooling represents a real next gate or just another
name for the closed route-excluded branch.

## Evidence Reviewed

- The 2026-06-03 periodic strategy review.
- Route-excluded source, route replay, prior-bootstrap, candidate-evidence, and
  background-refresh outcomes.
- Current-batch prior-target tooling and smoke evidence.
- The current `RESEARCH_STATE.md` next-experiment guidance.
- Memory retrieval for repeated route-excluded/shared-prior variants.

## Assessment

Current-batch prior targets are close to the closed shared-prior patch branch,
but they are not identical to the failed variants:

- They are not route replay: the loss is applied to examples already in the
  live training batch, not to a separately sampled pool.
- They are not prior bootstrap: no pseudo-target is written into the
  prompt-memory table.
- They are not candidate-evidence or background refresh: no extra candidate
  scoring is performed by the new path.

That distinction matters because the repeated failure pattern was that shared
prior information stayed outside the current source-training target path or got
copied back into prompt memory too late. Current-batch prior targets directly
ask whether the live batch can use the shared prior as a target source without
adding prompt-memory entries.

But the tooling does not yet solve the strategic problem. It still depends on
a prior trained from prompt-memory targets, so it has not removed the
answer-derived target source. It is a narrow target-formation gate, not a
non-prescriptive credit-assignment method.

## Decision

Allow exactly one real source gate for current-batch shared-prior target supply.
The gate should be framed as:

- Does direct current-batch prior-target loss improve heldout prompt quality
  and excluded-route quality when direct target discovery is reduced or absent
  for a route?
- Does it do so without writing prompt-memory pseudo-targets or adding extra
  candidate scoring beyond the source recipe?
- Does it meet the predeclared source-quality gate before any trusted handoff?

If that source gate misses, close this branch as another insufficient
shared-prior target-supply path. Do not run weight, confidence, route, cadence,
or seed ladders as novelty.

## Anti-Rerun Rules

Do not restart these closed branches under current-batch naming:

- Route replay weight ladders.
- Prior-bootstrap confidence/train-accuracy/cap ladders.
- Candidate-evidence weight/timing ladders.
- Background-refresh batch/every/weight ladders.
- Cap/fraction/window/validation hidden-size polishing.
- Route-heldout diagnostic route/seed ladders.

The only legitimate follow-up after a miss is a more global mechanism: joint
target learning across routes/calculators, amortization whose update count does
not scale linearly with calculator count, or a less-prescriptive credit signal
that bypasses answer-derived candidate scoring.

## Bottom Line

Current-batch prior targets are worth one real source gate because they are a
direct non-memory target-supply mechanism. They are not enough to revive the
local route-excluded patch family. Future agents should treat the gate as a
binary strategic test: pass it and proceed to trusted handoff; miss it and move
to genuinely shared/global target formation or less-prescriptive credit.
