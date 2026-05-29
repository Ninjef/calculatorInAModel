# Scheduled Source Geometry Review

Date: 2026-05-29

## Why This Review Exists

The recent branch introduced a direct source-acquisition objective for additive
handoff geometry:

```text
--additive-forced-true-loss-weight
--additive-forced-true-start-step
```

The user asked for periodic reviews to prevent loops. This review decides what
the scheduled source-geometry branch means and what should stop.

## What Changed

The project now has evidence that source acquisition can be trained for
additive readout geometry directly, not just selected afterward:

- Small `0..9` gate: scheduled forced-true auxiliary preserved source
  acquisition while improving geometry.
- Full `0..19` 200-step gate: scheduled source nearly tied baseline source
  accuracy but improved trusted 600-step handoff from `0.2525` to `0.4150`.
- Full `0..19` 800-step scheduled source: step-600 source checkpoint reached
  `0.7725` final eval under standalone 600-step frozen-policy additive
  handoff.

This is a real branch result, not a selector-only proxy result.

## What Should Stop

- Do not rerun the small `0..9` scheduled gate.
- Do not rerun the seed-13 `0..19` 200-step baseline-vs-scheduled handoff.
- Do not rerun the seed-13 scheduled source `200/400/600/800` geometry ladder
  or the step-600 vs step-800 handoff comparison.
- Do not treat forced-result geometry as a replacement for the standalone
  600-step handoff gate. Step `800` had perfect forced-result geometry but
  worse handoff than step `600`.
- Do not pivot back to cheap checkpoint selectors unless they are explicitly
  validated against fresh-family 600-step handoff outcomes.

## What Deserves Compute

Priority:

1. Continue/readout from the scheduled step-600 handoff lineage to test whether
   this source can clear the high non-bottleneck gate.
2. Replicate scheduled source acquisition on a fresh seed if stability becomes
   the explicit question.
3. Add a behavior-gated or policy-retention anchor only if longer runs show
   source-policy drift or if continuation/readout exposes degradation.

Lower priority:

- More start-step sweeps. A minor step-50/step-75 tweak is not a new mechanism
  unless tied to a predeclared behavior gate.
- More forced-result geometry-only ranking.

## Are We Closer?

Yes, but not done.

Closer:

- We have a training pressure that improves additive non-bottleneck transfer
  geometry during source acquisition.
- The source-geometry objective produced a large handoff improvement on the
  real full grid.

Still missing:

- A full non-bottleneck gate clear from this source branch.
- A scalable/non-prescriptive method. The current branch is still
  prescriptive and still uses hard assignment plus true-result auxiliary.
- Fresh-seed stability.

## Next Direction

Use the scheduled step-600 source/handoff lineage as the next handoff candidate
for continuation/readout. Keep actual handoff verification in the loop.

## Addendum: Continuation/Readout Result

The scheduled step-600 handoff lineage was tested with the standard downstream
recipe:

- 800-step frozen-policy continuation: `0.7775` final eval.
- 600-step policy-backbone-frozen readout after continuation: `0.8175`.
- Extra 1000 stable-policy readout steps: `0.8475`.

Controls stayed low, so the branch remains calculator-dependent, but it missed
the high non-bottleneck gate. Learned calculator accuracy stayed around
`0.5391`, unlike earlier positive continuation/readout lineages with much
higher learned calc.

Updated steering: do not spend more compute on this exact continuation/readout
chain. The scheduled source branch now needs source-policy accuracy or
retention pressure while preserving the additive geometry gain.

## Addendum: Low-LR Source Recovery

The source-policy diagnosis was correct. Continuing the scheduled step-600
source checkpoint gently, instead of extending the high-pressure source run,
restored learned calculator quality while preserving enough additive geometry:

- A 5-step smoke at the original LR `0.003` collapsed source normal from
  `0.5800` to `0.1700`.
- A 30-step recovery at LR `0.0003` with forced-true weight reduced to `0.1`
  raised source normal/calc to `0.7950`.
- The recovered source reached `0.8425` final eval under the trusted 600-step
  frozen-policy handoff.
- Standard 800-step continuation reached `0.8900` final eval and `0.9175`
  best snapshot.
- Standard 600-step readout reached `0.9320` final eval. A zero-step
  counterfactual eval of the readout final checkpoint measured normal
  `0.9225`, injection-zero `0.0300`, forced-random `0.0325`, oracle `0.9050`,
  and learned calc `0.7925`.

Updated steering: this branch now has a high non-bottleneck gate clear, but it
is still prescriptive and checkpointed. The next useful work is fresh-seed
replication or turning the gentle recovery into an automatic late-source phase,
with the 600-step handoff plus continuation/readout gates kept as the arbiter.

## Addendum: Fresh-Seed Recovery Replication

The gentle recovery recipe replicated on a fresh scheduled source seed:

- Seed-14 scheduled source training reached step-600 source eval `0.6675`,
  with forced-true additive loss `0.2202`.
- The same 30-step recovery recipe, LR `0.0003` and forced-true weight `0.1`,
  raised source eval to `0.8850` and reduced forced-true loss to `0.1433`.
- The trusted 600-step frozen-policy additive handoff from the recovered
  checkpoint reached `0.9600` final eval and `0.9650` step-600 snapshot.
- Final snapshot controls: injection-zero `0.0850`, forced-random `0.0875`,
  oracle `0.9850`, learned calc `0.8700`.

Updated steering: the low-LR recovery phase is no longer a one-seed artifact.
The zero/random controls on seed 14 are higher than seed 13, but still far
below normal. Further value comes from automating the late-source transition
or explicitly testing stability on another seed, not from rerunning this exact
seed-14 chain.

## Addendum: Automated Late-Source Recovery

The manual checkpoint relaunch was replaced with an in-run late-source recovery
switch:

- `overfit_one_batch.py` now accepts a non-default late-source recovery phase
  that can lower optimizer LR and override the forced-true additive weight at a
  configured step.
- On seed 14, one 630-step source run switched at step `600` to LR multiplier
  `0.1` and forced-true weight `0.1`.
- The automated source reached final eval `0.8775` versus `0.8850` for the
  manual relaunch.
- The trusted 600-step frozen-policy handoff reached `0.9400` final eval and
  `0.9475` step-600 snapshot, versus `0.9600`/`0.9650` for manual relaunch.
- Final handoff snapshot controls: injection-zero `0.0800`, forced-random
  `0.0775`, oracle `0.9650`, learned calc `0.8725`.

Updated steering: the late-source phase is now automatable. It is still
fixed-step and prescriptive, so the next question is adaptive transition
criteria or a return to scalable/non-prescriptive assignment mechanisms, with
the trusted handoff/readout gates retained.

## Addendum: Adaptive Recovery Trigger

The first corrected adaptive transition is mixed-positive:

- Added source-metric trigger flags for late-source recovery and fixed the
  adaptive path so it lowers both LR and forced-true auxiliary weight after the
  trigger.
- On seed 14, `result_policy_argmax_result_accuracy >= 0.65` with min step
  `500` triggered at step `528`.
- The adaptive source final eval was only `0.8250`, but its step-600 source
  snapshot was `0.8825` with zero/forced-random `0.0425`/`0.0200`.
- The trusted 600-step frozen-policy handoff reached `0.9850` final eval and
  `0.9775` step-600 snapshot, beating the fixed step-600 automated handoff
  (`0.9400` final).
- Handoff controls worsened: injection-zero and forced-random were both
  `0.1325`, versus `0.0800`/`0.0775` for the fixed-step run.

Updated steering: source final eval is not enough to judge transition quality,
and the simple adaptive trigger deserves replication or smoothing before it
becomes the default. Keep using trusted handoff controls as the arbiter; do not
rerun the same seed-14 threshold as novelty.

## Addendum: Fresh Adaptive Trigger Replication

The seed-14 adaptive result did not replicate with raw source-accuracy
thresholding:

- On fresh seed 17, `result_policy_argmax_result_accuracy >= 0.65` with min
  step `500` never fired.
- The no-trigger source ended at `0.6100`; its trusted 600-step handoff reached
  only `0.6825` final eval / `0.6925` step-600 snapshot, with injection-zero
  `0.0400`, forced-random `0.0500`, and learned calc `0.6075`.
- A matched fixed step-600 recovery control improved source final to `0.7450`
  and handoff to `0.7675` final eval / `0.7850` snapshot, with injection-zero
  `0.0500`, forced-random `0.0375`, and learned calc `0.7350`.
- The fixed-step control still missed the high non-bottleneck gate, so this
  seed is harder overall, but the adaptive threshold also failed to activate a
  useful late phase.

Updated steering: raw source-argmax thresholding is not validated. The next
adaptive transition should use smoothing, patience, a conjunction with
geometry/loss, or another trigger family; otherwise return to scalable
assignment work rather than tuning this exact threshold.
