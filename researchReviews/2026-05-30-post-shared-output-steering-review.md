# 2026-05-30 Post-Shared-Output Steering Review

## Why This Review Exists

The user asked for periodic reviews that prevent agents from re-running old
ideas or getting absorbed by minor variants. The shared-output branch just
closed a real many-calculator scaling question, but it also risks becoming a
new local loop.

This review decides whether to keep spending mainline compute on routed/shared
output scaling, or pivot back to the central training-method problem.

## What Changed

The many-calculator branch now has a sharper boundary:

- Four routed calculator hooks can train and transfer when they use cloned
  semantic output projections: source final `0.9950`, trusted handoff final
  `1.0000`, low injection-zero and forced-random controls.
- Active-only routing removes the known all-hooks-forward waste for routed
  batches.
- Shared output projection removes cloned-output parameter growth and still
  trains the four-hook source.
- The shared-output trusted handoff misses despite near-perfect learned
  calculator accuracy: first handoff `0.7625` final / `0.7800` step-600
  normal; matched delayed-margin rerun `0.7475` final / `0.7225` step-600
  normal.
- A state-dict/tying audit ruled out a loading/forward-equivalence bug.
- Follow-up with a new source mechanism changed the boundary: online hard
  memory plus additive semantic distillation cleared four-hook shared-output
  source and trusted handoff (`1.0000` final / step-600 normal, low controls).
- That pass replicated on CLI seed `7` / effective seed `9`, the fresh
  handoff-sensitive lineage where the single-hook semantic-distilled source had
  missed trusted handoff.

The last audit also found a config mismatch in the first A/B
(`additive_forced_margin_start_step=50` for cloned, default `0` for shared),
then showed that matching the delay does not rescue tied-output handoff.

## Belief Changes

Routed calculators are no longer the main unknown. They can train, can be
active-only, and can use a shared semantic output projection at source time.

The shared-output failure was not architectural inevitability; it was a
transfer/readout-geometry failure for the hard-assignment source. The remaining
question is whether a source objective can create a handoff-friendly shared
semantic interface across seeds, ranges, and prompt regimes. The thesis
bottleneck remains scalable, less-prescriptive credit assignment into the
calculator-query policy.

## What Should Stop

Do not treat these as novel:

- More same-recipe shared-output source630 plus handoff600 runs.
- Shared-output continuation runs without a new readout/source mechanism.
- Delayed-margin or start-step reruns of the same tied-output recipe.
- Repeating the same semantic-distilled four-hook shared-output seed as
  novelty.
- More op19 semantic-distilled four-hook shared-output seed repeats as
  mainline work; two seeds now clear.
- More op19/op29 topk8+unique24 replications.
- More forced-margin recovery knob changes.
- Cheap source selectors or short handoff proxies that do not replace the
  trusted handoff gate across fresh families.

## What Deserves Compute

Mainline compute should move toward less-prescriptive credit assignment:

- answer-derived target construction that avoids specifying the true useful
  calculator result;
- estimators that reduce or replace forced-result enumeration;
- streaming or evolving-state validation for target construction;
- uncertainty-aware candidate scoring only if it changes the estimator, not
  just the sample count;
- a concrete Stage 0/Stage 1 gate showing from-scratch calculator-result lift
  above known failed baselines.

Shared-output work deserves compute only when it tests a new generalization
axis or mechanism: streaming/fresh-prompt memory, larger ranges, handoff-aware
source shaping, route-aware downstream readout, or a predeclared tied-output
transfer-geometry objective validated by the trusted handoff gate.

## Are We Closer?

Yes, but mostly by eliminating confusion.

The architecture and staged transfer story are stronger: many routed
calculators can be trained and active-only routing makes the implementation
more scalable. The semantic-distilled routed result shows that shared output
can transfer when the source objective teaches a more handoff-friendly result
interface.

The full goal is still not met because the best methods are prescriptive and
depend on hard assignment, candidate scoring, pretrained decoders, and frozen
transfer. The next strategically valuable progress is not another routed
scaling variant. It is a changed credit-assignment method.

## Steering Decision

Pause same-recipe shared-output scaling as a mainline. Future agents should
start from:

```text
less-prescriptive credit assignment first;
shared-output only with streaming/range stress or a new transfer-geometry mechanism.
```

If the next proposed experiment cannot say how it reduces prescriptiveness,
replaces/approximates forced-result enumeration, or introduces a new
handoff-geometry mechanism, write a review instead of running it.
