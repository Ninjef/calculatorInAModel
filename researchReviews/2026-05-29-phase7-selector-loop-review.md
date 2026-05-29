# 2026-05-29 Phase 7 Selector-Loop Review

## Why This Review Exists

This review follows the earlier Phase 7 strategy review on the same date. The
next cluster of work produced useful evidence, but it also circled back to a
familiar pattern: trying cheaper and cheaper source selectors after the project
had already identified credit assignment and transfer geometry as the main
strategic bottlenecks.

The point of this memo is to keep future agents from treating every failed
proxy as an invitation to invent a nearby proxy.

## What Changed

The staged non-bottleneck recipe is more real than before, but still not a
final scalable method.

- Fresh `src6` nearly cleared the frozen-policy handoff gate (`0.8975`), then
  cleared after 800-step continuation and 600-step readout (`0.9850` final).
- Fresh `src7` confirmed that a 600-step selector can pick the best available
  source checkpoint, but the reduced recipe still missed gate (`0.8825`).
- No-decay entropy/diversity source acquisition prevented source collapse and
  produced a seed-9 source that cleared after continuation/readout (`0.9575`).
- The same no-decay recipe on seed 10 produced high bottleneck calculator
  accuracy (`~0.90`) but transfer-hostile geometry: handoff, direct readout,
  and continuation stayed far below gate.

The selector story also changed:

- 600-step standalone frozen-policy handoff remains the trusted source gate.
- 500-step selection failed on fresh `src6` and an in-training seed-11 probe.
- Forced-result geometry can flag hostile seed-10-style sources, but it fails
  as a reliable checkpoint selector across normal families.
- 25/50/100-step downstream loss slope and a simple ridge selector over early
  handoff traces failed to beat raw handoff exact traces.

## Belief Changes

The bottleneck has moved one level upstream within the staged transfer branch.

It is not enough to acquire a strong bottleneck calculator policy, and it is
not enough to select from snapshots using source accuracy or cheap geometry
proxies. The source objective can improve learned calculator accuracy while
making additive readout geometry worse, as seed 10 showed.

The branch now needs source acquisition that directly shapes downstream
handoff/readout geometry, or a genuinely different credit-assignment mechanism.
Selector-cost reduction is no longer a good default next step.

## Branches That Should Stop

Do not keep running local selector variants unless they beat the established
gate on fresh families.

Specifically, stop treating these as mainline progress:

- source normal/calculator accuracy selectors;
- 500-step selector claims without standalone 600-step confirmation;
- forced-result geometry as a sole selector;
- 25/50/100-step loss slope;
- frozen-state linear probes;
- simple ridge selectors over the current early-trace feature set;
- embedded 500-step in-training probe normal score as a selector.

These can remain diagnostics or logging signals. They should not decide source
checkpoint selection by themselves.

## Branches That Deserve Compute

Two classes still deserve serious work.

1. Source acquisition that optimizes for transfer/readout behavior directly.
   Examples: a source-time auxiliary based on actual short additive handoff
   exact, continuation slope, injection-to-answer geometry, or another target
   that is validated against the standalone 600-step handoff gate.
2. A new credit-assignment family, especially target propagation or local
   targets, with a small feasibility gate before long runs.

A selector experiment is only worth running if it is part of class 1 and
changes training pressure or scalability, not merely ranking already produced
checkpoints with another cheap metric.

## Are We Closer To The Goal?

Yes, but the progress is diagnostic and architectural rather than a final
training method.

The project now has stronger evidence that calculator-dependent non-bottleneck
use can be achieved by staged transfer and protected policies. It also has a
clearer negative: strong bottleneck calculator use does not imply transferable
additive geometry.

The missing result remains the same: scalable, non-prescriptive training from
scratch that assigns credit to calculator-query policy choices.

## What Counts As Success Next

A useful next result should meet one of these standards:

- A source-training objective improves fresh-family 600-step handoff and later
  continuation/readout outcomes compared with the current no-decay source
  recipe.
- A target-propagation/local-target prototype passes a Stage 0 feasibility gate
  and produces early Stage 1 calculator-result lift above known failed
  answer-loss and shadow-gradient baselines.
- A lower-cost assignment approximation preserves the bottleneck hard
  improvement-assignment source result while reducing the need for full result
  enumeration.

## Steering Decision

Mainline work should stop optimizing checkpoint selectors as a standalone
branch. Keep standalone 600-step handoff as the source gate, use cheap geometry
only as logging or rejection evidence, and spend new experiment budget on
training objectives that make sources more handoff-readable in the first
place.
