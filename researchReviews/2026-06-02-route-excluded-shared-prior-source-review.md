# Route-Excluded Shared-Prior Source Review

Date: 2026-06-02

## Question

Have we moved from post-hoc route-heldout diagnostics to an actual source run
where some routed calculators receive no direct prompt-memory target discovery?

## Findings

- The route-exclusion training path is now real after the prompt-keyed plumbing
  fix. The op19 run filled only score-eligible memory (`223/223`) and reported
  excluded/update fractions, so the experiment is valid evidence.
- The shared numeric prior can partially supply the excluded route. Route 1
  reached `0.7304` snapshot calc, `0.8000` diagnostic calc, and `0.7391`
  heldout-route calc despite no direct prompt-memory updates.
- The source gate still missed. Final exact/calc was `0.7875`, best snapshot
  `0.8075`, heldout prompts `0.5625`, and prior heldout `0.5625`. Controls were
  causal, so this is not leakage, but it is not a trustworthy source for
  handoff.
- The branch should stop rerunning route-heldout diagnostics, op9 preflights, or
  the same op19 route-excluded source recipe. The new evidence points to a
  mechanism gap, not a need for another schedule/seed pass.

## Direction

The useful next work is a stronger shared/global target mechanism: route-balanced
or global prior replay, shared target discovery across calculators, explicit
global-prior training objectives, or a credit-assignment method that removes
per-route prompt-memory target tables and answer-derived candidate scoring.

## Evidence

- `aiAgentWorkHistory/phase7/2026-06-02-route-heldout-shared-prior-diagnostic.md`
- `aiAgentWorkHistory/phase7/2026-06-02-route-excluded-shared-prior-preflight.md`
- `aiAgentWorkHistory/phase7/2026-06-02-op19-route-excluded-shared-prior-source.md`
