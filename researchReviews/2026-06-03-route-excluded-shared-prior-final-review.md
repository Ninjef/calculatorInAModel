# 2026-06-03 - Route-Excluded Shared-Prior Final Review

## Scope

This review closes the current route-excluded tweak branch after the full
candidate-evidence source gate.

Reviewed evidence:

- Offline route-heldout numeric-prior diagnostic.
- Corrected route-excluded prompt-memory source.
- Extra route-weighted prior replay.
- Prior-bootstrap prompt-memory entries.
- Candidate-evidence prior updates.

## Findings

The route-heldout diagnostic was not misleading about structure: numeric priors
can share calculator target structure across routed hooks when trained
post-hoc on clean memory. The failure is the live source process.

All live op19 route-excluded variants missed the heldout source gate:

| Variant | Final eval | Best snapshot | Heldout prompts | Prior train/heldout |
| --- | ---: | ---: | ---: | ---: |
| No bootstrap | `0.7875` | `0.8075` | `0.5625` | `0.7781 / 0.5625` |
| Route replay w2 | `0.8175` | `0.8075` | `0.5750` | `0.7750 / 0.5750` |
| Prior bootstrap | `0.7700` | `0.7825` | `0.5625` | `0.7781 / 0.5625` |
| Candidate evidence | `0.7725` | `0.8000` | `0.5375` | `0.7156 / 0.5375` |

Candidate-evidence prior updates did fire (`32` updates over `1060` examples),
but prompt memory filled by step `50`, leaving little opportunity to reshape
the shared prior before the route-excluded source settled. Excluded route 1
heldout was `0.6522`, worse than the no-bootstrap/route-replay heldout route
value, and no trusted handoff was justified.

## Decision

Close the route-excluded tweak branch. Do not run route-replay weights,
bootstrap thresholds/caps, candidate-evidence weights/timings, or same-recipe
seed repeats as novelty. These are local pressure/timing changes around the
same per-route prompt-memory target system.

The next research move should change the target formation system itself:

- Shared/global target discovery across routes before per-route memory freezes.
- Joint target learning where routes update one shared target model rather than
  isolated prompt tables.
- A less-prescriptive credit signal that removes answer-derived candidate
  scoring and per-route prompt-memory tables.

The useful lesson is narrow but important: routed calculators can share numeric
target structure offline, yet the current live source acquisition path cannot
turn that structure into a heldout-generalizing route-excluded source.
