# Research State

Last updated: 2026-05-29

Maintenance rule: keep this file under about `200` lines. Replace stale
synthesis instead of appending history. Put older strategic snapshots in
`researchReviews/`, topic memories in `researchMemory/`, and raw evidence in
`factSheets/` or `aiAgentWorkHistory/`.

## Overarching Goal

Prove that a model can be trained from scratch to use a non-differentiable
calculator embedded inside its neural computation, in a way that is scalable
and works both when the calculator is a bottleneck and when a pure-neuron path
also exists.

The ideal method is non-prescriptive: it lets the model discover useful
calculator queries rather than telling it which query/result to request for
each problem.

## Current Bottom Line

The architecture is viable, but the central learning problem is not solved.

We have shown that calculator-use policies can be scaffolded, retained, and
transferred into a non-bottleneck model. We have not shown scalable,
non-prescriptive, answer-loss-only discovery of the calculator-query policy.

The current bottleneck is credit assignment into the calculator-query policy,
not downstream answer decoding and not calculator wiring. Within the staged
transfer branch, the current bottleneck is source acquisition for additive
handoff/readout geometry, not source calculator accuracy or another cheap
checkpoint selector.

## What Is Proven

- Correct calculator outputs or oracle calculator queries let downstream
  components solve the arithmetic task. This proves wiring, not the thesis.
- Identifiable/scaffolded calculator protocols can be taught and often retained
  after direct supervision or local targets are removed.
- Natural result-level bottleneck calculator policies can be trained with hard
  improvement-assignment style targets.
- A trained bottleneck calculator policy can be transferred into an additive
  non-bottleneck model if the policy is frozen or strongly protected.
- Some staged handoff/continuation/readout chains clear the non-bottleneck
  performance gate with low injection-zero and low forced-random controls.

## What Is Not Proven

- From-scratch answer-loss-only discovery of calculator use.
- A scalable method that avoids enumerating/scoring many calculator candidates
  during training.
- A non-prescriptive method that lets the model decide how to use calculator
  results without a teacher specifying the useful result/query.
- Robust non-bottleneck discovery when the model can bypass the calculator.
- A cheap proxy that reliably replaces actual 500/600-step additive handoff
  checks for source checkpoint selection.

## Strategic Bottleneck

The model needs a useful learning signal for upstream calculator queries.
Plain answer loss, score-function estimators, expected answer-loss gradients,
decoder calibration, simple direct feedback, and simple learned shadow-gradient
variants have not produced reliable discovery.

The strongest current positives depend on scaffolding or candidate scoring.
That means the project is beyond "can the architecture work?" and is now at
"can we find a scalable credit-assignment mechanism?"

## Current Best Recipe

The best known non-bottleneck recipe is staged:

1. Train a bottleneck source policy with hard improvement assignment and
   stabilization.
2. Select or verify source checkpoints with actual additive handoff behavior.
   For fresh source families, prefer a standalone 600-step frozen-policy
   handoff gate; 500-step and embedded probes are logging/triage only until
   reconfirmed by the 600-step gate.
3. Transfer into additive non-bottleneck mode.
4. Freeze or protect the calculator policy.
5. Train downstream/readout and, when needed, continuation/readout stages.

This can work, but it is not the final answer because it is costly,
checkpoint-sensitive, and still prescriptive.

## Active Strategic Bet

The next worthwhile work should change the credit-assignment mechanism or train
source policies directly for transfer/readout geometry.

Active directions:

- Source acquisition optimized against actual handoff/readout geometry, not
  just source answer accuracy or cheap selector scores. A first
  forced-true additive readout auxiliary shows this can shape transfer
  geometry. The naive always-on version competes with source policy
  acquisition, but a delayed start fixed that tradeoff in a small gate and
  improved full-grid 600-step handoff from `0.2525` to `0.4150` at matched
  200-step source checkpoints; extending scheduled source training to step
  `600` raised final handoff to `0.7725`, but continuation/readout plateaued
  below gate at `0.8475` with learned calc around `0.5391`. A gentle
  low-LR source recovery from that step-600 checkpoint fixed the immediate
  source-policy bottleneck: source calc rose to `0.7950`, 600-step handoff
  improved to `0.8425`, and continuation/readout cleared the high
  non-bottleneck gate at `0.9320` with low zero/random controls. A fresh seed
  replicated the recovery effect and cleared the handoff gate directly:
  seed-14 scheduled source step `600` recovered from `0.6675` to `0.8850`
  source eval, then reached `0.9600` final frozen-policy handoff with
  injection-zero/forced-random around `0.085`. Folding the same late phase
  into one automated run on seed 14 preserved the result: final source eval
  `0.8775` and 600-step handoff `0.9400`, with low zero/random controls.
  A simple source-accuracy adaptive trigger is not yet robust: on seed 14,
  `argmax_result_accuracy >= 0.65` fired at step `528` and improved handoff to
  `0.9850` final eval but with higher zero/random controls (`0.1325`); on a
  fresh seed it never fired, source final was `0.6100`, and handoff reached
  only `0.6825` while a fixed step-600 control reached `0.7675`. A smoothed
  forced-true-loss trigger with patience fired at step `509` and improved the
  hard-seed handoff to `0.8025`, but still missed the high gate. A hard
  conjunctive gate requiring source accuracy `>=0.70` never fired on the same
  seed and fell back to the weak no-recovery handoff (`0.6825`).
- A genuinely different credit-assignment family such as target propagation or
  local targets. Exact `policy_reweighted_t1` is positive and survives
  answer-only retention, but full enumeration is not scalable. Naive sparse
  sampling and low-loss neighborhoods failed; a new replay-memory proposal is
  the first useful approximation. With cached per-prompt losses, `u8_m24`
  beats raw uniform `u32` at 200 steps (`0.5900` vs `0.3350` exact calc) and
  retains `0.8600` calc / `0.8750` sampled normal after an 800+200 gate. A
  lower-budget `u2_m30` point improves the 200-step gate to `0.6025` calc /
  `0.6016` sampled normal, but retains less strongly (`0.7850` calc /
  `0.7656` normal). Light cached-candidate rescoring ties `u2_m30`; heavier
  rescoring hurts.
- Lower-cost assignment is useful only when it changes scalability, not merely
  proxy selection.

## Paused Or Deprioritized Branches

These branches should not continue without a new mechanism:

- Oracle-only calculator success.
- Generic target-off retention after scaffolding.
- Vanilla result-space REINFORCE and ordinary expected answer-loss gradients.
- Decoder calibration alone.
- Simple output-projection direct feedback.
- Fixed fit-once linear shadow feedback.
- Simple online MLP shadow-gradient variants that only tweak normalization,
  validation selection, dropout, or loss shape.
- Hand-coded local-target candidate proposal variants, including raw
  uniform/top-k sparse sampling and simple low-loss neighborhood expansion.
  Continue local targets only with a learned proposal, estimator correction, or
  different target construction.
- Cheap source selectors based on frozen-state readout, forced-result geometry,
  25/50/100-step loss slope, simple ridge over early traces, or 500-step
  embedded probe normal score alone. Use these only for logging, rejection
  warnings, or hypothesis generation unless validated against fresh-family
  600-step handoff outcomes.
- Source accuracy as a source-checkpoint selector.
- Slight weight/seed/length changes to already failed source-stabilization
  recipes unless tied to a new transfer-geometry objective.

## Next 1-3 Experiments

1. Stress-test replay-memory beyond fixed-grid transduction: reset/age memory,
   streaming/non-exhaustive prompts, or generalized proposals. Simple cached
   rescoring did not fix retention.
2. In parallel, keep source objectives aimed at actual handoff/readout geometry,
   not one-metric recovery triggers or cheap selectors.
3. If trying to reduce hard-assignment cost, state the scalability hypothesis
   up front and compare against the exact-grid assignment ceiling rather than
   only against prior cheap selectors.

Do not run another local selector/proxy experiment unless it is explicitly
designed to replace a named compute bottleneck and has a predeclared validation
against fresh-family 600-step handoff outcomes.

## What Would Change Our Mind

- A from-scratch training run where calculator-result accuracy rises because
  of answer-derived learning, not direct oracle/query supervision.
- A non-bottleneck run where normal accuracy is high, injection-zero and
  forced-random controls stay low, and the calculator policy was not simply
  frozen from a prescriptive source.
- A scalable approximation to hard improvement assignment that preserves the
  bottleneck source result while avoiding full result-class enumeration.
- A new credit-assignment method that passes a local feasibility gate and then
  produces early Stage 1 lift above known failed baselines.
- A source-training objective that predicts or improves downstream handoff
  behavior across fresh seeds, not just within already studied lineages.

## Review Cadence

Write or update a `researchReviews/` memo after any of the following:

- 5-10 new experiments.
- A branch produces a clear family-level negative.
- A branch produces a result that changes the active strategic bet.
- The next proposed task feels like another local variant of a paused family.

The review must answer what changed, what should stop, what deserves compute, and whether the project is closer to the overarching goal.
