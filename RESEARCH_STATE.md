# Research State

Last updated: 2026-05-29

Maintenance rule: keep under about `200` lines. Replace stale synthesis; move
older context to reviews, memories, fact sheets, or work logs.

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
  just source answer accuracy or cheap selector scores. Delayed forced-true
  additive readout pressure plus late low-LR recovery is a strong staged
  recipe: manual recovery cleared continuation/readout (`0.9320`), fresh-seed
  recovery reached `0.9600` trusted handoff, and one-run automated recovery
  reached `0.9400`. Simple source-accuracy and forced-loss triggers are not
  robust enough yet, and cheap selector/proxy work is paused unless validated
  against fresh-family 600-step handoffs. One-negative forced-margin is a
  useful source auxiliary: the early 200-step gate reached `0.6600` handoff,
  longer unrecovered sources reached only about `0.73-0.74`, and manual low-LR
  recovery raised handoff to `0.8700`. Folding margin recovery into the source
  run now replicates strongly on a fresh seed: source step `600->630` improved
  calc `0.5825->0.8825`, final source eval was `0.8700`, and trusted
  frozen-policy handoff reached `0.9875` final / `0.9800` step-600 normal with
  injection-zero `0.0156-0.0250` and forced-random `0.0938`. This is useful
  evidence for staged transfer, but it remains prescriptive because it uses
  hard assignment and true-result contrastive forcing.
- A genuinely different credit-assignment family such as target propagation or
  local targets. Exact `policy_reweighted_t1` is positive and survives
  retention, but full enumeration is not scalable. Simple proposal
  approximation is now paused after sparse/adaptive, replay-memory, corrected,
  online learned, and pretrained learned variants failed scalability stress.
  Continue only with a different estimator, different target construction, or
  explicitly streaming/generalizing learned proposal.
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
  uniform/top-k sparse sampling, low-loss neighborhoods, and fixed/prompt-keyed
  replay memory. Continue local targets only with a learned proposal, estimator
  correction, or different target construction.
- Cheap source selectors based on frozen-state readout, forced-result geometry,
  25/50/100-step loss slope, simple ridge over early traces, or 500-step
  embedded probe normal score alone. Use these only for logging, rejection
  warnings, or hypothesis generation unless validated against fresh-family
  600-step handoff outcomes.
- Source accuracy as a source-checkpoint selector.
- Slight weight/seed/length changes to already failed source-stabilization
  recipes unless tied to a new transfer-geometry objective.

## Next 1-3 Experiments

1. Do not run more simple local-target proposal variants; continue local
   targets only via estimator/target-construction changes or a learned proposal
   with an explicit streaming/generalization validation objective.
2. Keep source objectives aimed at actual handoff/readout geometry,
   not one-metric recovery triggers or cheap selectors.
3. If staying in forced-margin, do not rerun the seed-15 manual recovery or
   seed-16 automated recovery gates as novelty. The next question must be
   broader stability/scale or a less-prescriptive target-construction bridge.
4. If reducing hard-assignment cost, state the scalability hypothesis
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

Write or update a `researchReviews/` memo after 5-10 new experiments, a clear
family-level negative, a strategic-bet change, or when the next task feels like
another local variant of a paused family. The review must answer what changed,
what should stop, what deserves compute, and whether the project is closer to
the overarching goal.
