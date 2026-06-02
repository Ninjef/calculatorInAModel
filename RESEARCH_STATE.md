# Research State (Last updated: 2026-06-02)
Keep near `200` lines; move stale context to reviews, memories, fact sheets, or work logs.
## Overarching Goal
Prove that a model can be trained from scratch to use a non-differentiable
calculator inside its neural computation, in a scalable way that works both
with a calculator bottleneck and when a pure-neuron path also exists.

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

The best known non-bottleneck recipe is staged: train a bottleneck source,
verify with an actual standalone 600-step frozen-policy additive handoff for
fresh families, transfer into additive mode, freeze/protect the calculator
policy, and train downstream/readout stages. Shorter or embedded handoff probes
are logging/triage until reconfirmed by the 600-step gate.

This can work, but it is costly, checkpoint-sensitive, and still prescriptive.

## Active Strategic Bet

The next worthwhile work should change the credit-assignment mechanism or train
source policies directly for transfer/readout geometry.

Active directions:

- Source acquisition optimized against actual handoff/readout geometry, not
  source accuracy or cheap selector scores. Forced-true/forced-margin with
  recovery is the staged benchmark: it clears multiple op19/op29 handoff gates
  when source/readout capacity is sufficient, and op39 is causal but costly and
  sub-perfect. Treat it as the benchmark to beat, not the scalable or
  non-prescriptive recipe.
- Target propagation/local targets are now a ceiling and diagnostic, not the
  current scalable mainline. Exact `policy_reweighted_t1` is positive and
  survives retention, but full enumeration is not scalable. Simple
  approximations are paused after repeated scalability or Stage 1 failures.
  Continue only with a materially new target construction or streaming/
  heldout-generalization gate.
- Less-prescriptive answer-derived target construction. The old full-grid
  result-boundary target source transfers causally into the trusted additive
  frozen-policy gate (`0.8825` final, `0.8425` step-600 normal, zero-injection
  `0.0000`, learned calc `0.9922`). True-result forced-margin is not strictly
  required, but full enumeration and frozen transfer remain unsolved issues.
  Static approximations are paused: critics/proposals are costly or state-local,
  soft/regret/sampled hard-best targets are weak, and simple online calibration
  is partial. Online hard memory plus additive semantic distillation is the
  sparse source lead: two single-hook op19 seeds reached `1.000` source/calc
  with capped forced evals (`76.8k-86.4k`). Handoff is source-geometry
  sensitive: one single-hook source cleared trusted handoff across two seeds; a
  fresh source missed (`0.647` / `0.632`) despite calc `1.000`. The same
  mechanism clears four-hook left-routed shared-output source/handoff on the
  original and fresh op19 seeds and an op29 range stress (`1.000` final, low
  controls), unlike prior shared-output hard-assignment misses. Prompt-keyed
  streaming minibatch memory now also clears op19 when the update budget is
  exposure-matched: batch64 for 800 steps undertrained (`0.6325` final), but
  batch64 for 5000 steps reached `1.000` source/calc and trusted handoff
  `1.000` with low controls while freezing memory after `173,568` forced evals.
  A 20% heldout-prompt split exposed the limit: train prompts reached `0.997`
  calc/exact, but heldout prompts only `0.0875`. Integrated numeric-prior replay
  with full-memory prior fitting fixes that heldout gate: source train
  `1.000`, heldout `0.9125`, overall `0.995`, and trusted handoff `1.000` with
  low controls. Fitting every other step preserves the result and clears handoff
  (`0.9875`) with half the prior updates; every `10` steps underfits and drops
  heldout to `0.7625`. Sustained convergence (`1.0` train accuracy for `100`
  fits) preserves source/handoff and cuts updates to `1889`; first-`1.0`
  stopping and random half-memory fits underfit (`0.875`/`0.8125` heldout).
  Target-stratified half-memory prior fitting is the first structured coreset
  positive: source overall `0.9900`, heldout `0.9375`, low heldout controls,
  forced-result evals `67,584`, and trusted frozen-policy handoff `0.9975`
  with diagnostic calc `1.0000` and low final controls. It preserves the gate
  but still uses `2501` prior updates. A validation-heldout stop that removed
  20% of memory from fitting reduced updates only slightly (`2359`) and missed
  heldout (`0.8625`), so do not run validation threshold/patience ladders.
  Eval-only validation stopping fits all memory entries while using a split
  only for stopping; on op19 it cut prior updates to `1613-1784` and cleared
  trusted handoff, but forced-result evals rose when prompt memory filled late.
  At op29, constant target-stratified batch `160` failed even with h128:
  heldout was only `0.8444`/`0.8611`, while post-hoc full-memory h128 recovered
  `0.9278`. The new post-memory-fill full-refresh mechanism changes the fit
  dynamics and clears the op29 source/handoff gate: source overall `0.9822`,
  train `0.9972`, heldout `0.9167`, prior updates `2755`, forced evals
  `294,912`, and trusted frozen-policy additive handoff `900/900 = 1.0000`
  with diagnostic calc `0.9922`, injection-zero `0.0000`, forced-zero
  `0.0000`, and forced-random `0.0078`. This is a real range-scaling positive
  for fit dynamics, but not yet the scalable recipe: it adds `2500`
  full-memory refresh updates after memory fill. Letting the existing
  validation stop end that refresh early cut prior updates to `1140` but missed
  heldout (`0.8167`), so simple early-stop ladders are not enough.
- Lower-cost assignment is useful only when it changes scalability; uniform
  sampling, fixed refresh, and unique-uniform sampling are insufficient.
  Topk8+unique24 clears op19/op29 staged gates but still scores candidates.
  Routed execution is now active-only, shared output removes cloned-output
  parameter growth, and semantic-distilled online hard memory is the first
  shared-output routed handoff family to replicate, pass op29, train under
  stochastic minibatches with matched exposure, and pass a heldout-prompt
  source/handoff gate with numeric-prior replay.

## Paused Or Deprioritized Branches

- Oracle-only calculator success.
- Generic target-off retention after scaffolding.
- Vanilla result-space REINFORCE and ordinary/rank expected answer-loss gradients.
- Decoder calibration alone.
- Simple output-projection direct feedback.
- Fixed fit-once linear shadow feedback.
- Simple online MLP shadow-gradient variants that only tweak normalization,
  validation selection, dropout, or loss shape.
- Hand-coded local-target candidate proposal variants, including raw
  uniform/top-k sparse sampling, low-loss neighborhoods, fixed/prompt-keyed
  replay memory, imputed sparse targets, simple learned proposals, and sparse
  pairwise preferences. Continue local targets only with a materially different
  estimator/target construction or predeclared streaming/generalization gate.
- Cheap source selectors based on frozen-state readout, forced-result geometry,
  25/50/100-step loss slope, simple ridge over early traces, or 500-step
  embedded probe normal score alone. Use these only for logging, rejection
  warnings, or hypothesis generation unless validated against fresh-family
  600-step handoff outcomes.
- Source accuracy as a source-checkpoint selector.
- Slight weight/seed/length tweaks unless tied to a new transfer objective.

## Next 1-3 Experiments

1. Reduce the cost of the op29 full-refresh positive without losing the heldout
   source and trusted handoff gates: staged full refresh then coreset replay,
   coverage-aware/proportional fitting, or a smarter stop/freeze transition.
   Do not rerun op29 batch160, hidden-size bumps, random fit-batch ladders,
   validation threshold/patience ladders, simple early full-refresh stopping,
   or the same full-refresh pass as novelty.
2. Keep source objectives aimed at actual handoff/readout geometry,
   not one-metric recovery triggers or cheap selectors.
3. Any further forced-margin, assignment-cost, or result-boundary transfer work
   must stress a new thesis-relevant axis: many-calculator cost, cheaper
   candidate scoring, removal of hard assignment/true-result forcing, or a
   changed estimator. Do not rerun completed op19/op29/op39 paths or static
   critic/proposal/soft/regret-set variants.

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
- A source-training objective that improves downstream handoff behavior across
  fresh seeds, not just within already studied lineages.
