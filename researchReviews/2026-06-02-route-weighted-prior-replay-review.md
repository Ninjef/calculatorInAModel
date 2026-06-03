# Route-Weighted Prior Replay Review

Date: 2026-06-02

## Question

Was the route-excluded source miss caused mainly by too little prior replay on
the excluded route?

## Findings

- The new route-filtered replay mechanism is real: it samples selected routed
  hooks from the global prompt pool and adds prior-pseudo-target pressure without
  adding candidate scoring or prompt-memory updates for those routes.
- The full op19 route-excluded source with route 1 replay weight `2.0` improved
  final eval to `0.8175`, but best snapshot stayed `0.8075` and heldout prompts
  only rose to `0.5750`.
- Excluded route 1 did not materially improve: final snapshot and heldout route
  were both `0.7391`, and the 128-sample diagnostic remained `0.8000`.
- The prior remained weak (`0.7750` train, `0.5750` heldout), so extra route
  replay mostly amplified a weak/noisy teacher rather than fixing target
  sharing.

## Direction

Do not run route-weight ladders. The next useful move is a different shared
target mechanism: global/shared target discovery, prior training directly from
candidate evidence before hard memory freezes, or a less-prescriptive credit
signal that avoids per-route prompt-memory target tables.

## Evidence

- `aiAgentWorkHistory/phase7/2026-06-02-op19-route-excluded-shared-prior-source.md`
- `aiAgentWorkHistory/phase7/2026-06-02-route-weighted-prior-replay-source.md`
