# 2026-05-29 Periodic Review: Selector-Loop Steering

## Aim

Perform the requested periodic review for repeated experiment loops and stale
theories. The focus was the post-strategy-review Phase 7 cluster, where agents
kept exploring cheaper source-checkpoint selectors after the project had
already identified credit assignment and transfer-geometry source acquisition
as the strategic bottleneck.

## Inputs Reviewed

- `CLAUDE.md`
- `RESEARCH_STATE.md`
- `HYPOTHESIS_LEDGER.md`
- `researchReviews/2026-05-29-phase7-strategy-review.md`
- `researchMemory/phase7-direction-memory.md`
- post-review Phase 7 work logs for `src6`, `src7`, no-decay source
  stabilization, seed-10 geometry, additive geometry probes, short-slope
  probes, ridge trace selectors, and embedded in-training handoff probes.

The semantic memory search was also queried. The local server was unavailable,
so the script used the one-shot fallback. Top hits confirmed that the newest
memories already flag the 500-step embedded probe, forced-result geometry,
short-slope proxy, and source-accuracy selector as unreliable.

## Decision

Added a new zoom-out memo:

```text
researchReviews/2026-05-29-phase7-selector-loop-review.md
```

Updated the current strategic state and Phase 7 direction memory to make the
steering decision harder to miss:

- standalone 600-step frozen-policy additive handoff remains the trusted
  source gate for fresh families;
- 500-step, embedded 500-step, forced-geometry, short-slope, frozen-state, and
  simple ridge selectors are logging/triage only unless reconfirmed against
  fresh-family 600-step handoff;
- selector-cost reduction is not the default frontier;
- the next mainline work should change source-training pressure toward
  handoff/readout geometry, or move to target propagation/local targets.

## Anti-Rerun Note

Do not run another local selector/proxy variant merely because the previous
cheap proxy failed. A non-duplicative selector task must either beat the
standalone 600-step gate on fresh held-out source families or be part of a
source-acquisition objective that directly optimizes downstream
handoff/readout behavior.

## Verification

This was a documentation-only review. No experiment code changed.
