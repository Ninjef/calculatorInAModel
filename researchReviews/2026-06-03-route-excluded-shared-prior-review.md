# 2026-06-03 - Route-Excluded Shared-Prior Review

## Scope

Review the recent route-excluded shared-prior branch:

- Offline route-heldout numeric-prior diagnostic.
- Corrected prompt-memory route exclusion.
- Full op19 route-excluded source.
- Extra route-weighted prior replay.
- Prior-bootstrap prompt-memory source.

## Findings

The offline diagnostic was real but not sufficient. Numeric features can
generalize target structure across routes when trained post-hoc on clean memory:
heldout-route accuracies were `0.9333-0.9793`, and embedding controls failed.
That means there is shareable structure, not just route memorization.

The live source gate is the bottleneck. Route 1 can learn somewhat with no
direct target discovery, but the full op19 source misses: no-bootstrap final
`0.7875`, route-replay final `0.8175`, and prior-bootstrap final `0.7700`.
Heldout prompts remain around `0.5625-0.5750`, and excluded route 1 heldout
stays `0.7391`.

Post-hoc pressure on a weak prior is not the right lever. Extra replay only
increases pressure on model logits, and bootstrap writes route targets only
after the prior becomes barely adequate. Neither changes the shared-prior
formation problem.

Do not keep running small variants of this branch:

- No route-replay weight ladders.
- No bootstrap confidence/train-accuracy/cap ladders.
- No more short op9 route-excluded preflights.
- No route-heldout diagnostic route/seed ladders.

## Direction

The next route-scaling test needs a mechanism that changes how shared targets
are formed, not how an already-weak prior is replayed or copied:

- Train the shared prior on candidate evidence before prompt memory freezes.
- Learn shared/global targets jointly across routes rather than adding them
  after per-route memories are mostly settled.
- Replace answer-derived candidate scoring with a less-prescriptive credit
  signal that does not maintain per-route prompt-memory target tables.

The branch remains useful evidence: routed shared-output calculators can share
numeric target structure, but live source acquisition still depends too much on
per-route prompt-memory discovery.
