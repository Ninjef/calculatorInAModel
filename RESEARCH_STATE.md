# Research State (Last updated: 2026-05-31)
Maintenance rule: keep near `200` lines; move stale context to reviews, memories, fact sheets, or work logs.
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
  source accuracy or cheap selector scores. Delayed forced-true plus late
  low-LR recovery is a strong staged recipe (`0.9320` continuation/readout,
  `0.9600` fresh-seed handoff, `0.9400` automated handoff), but simple
  source-accuracy/forced-loss triggers are not robust. One-negative
  forced-margin with late recovery is the current benchmark: manual recovery
  raised handoff to `0.8700`; automated recovery hit `0.9875` final /
  `0.9800` step-600 normal on one fresh seed; a second fresh seed cleared at
  `0.8975` / `0.9050`; and wider `n_embd=32`, `n_head=2` non-product and
  product-decoder op19 stresses both reached `1.0000` final / step-600 normal.
  A shallow-head op29 range stress missed (`0.8533`), low-LR recovery partly
  rescued it (`0.9067`), and `rhead64` cleared op29 on two seeds (`1.0000`).
  The first op39 `rhead64` stress was causal but costly and sub-perfect
  (`0.9475` handoff after checkpoint continuation). Treat forced-margin as a
  strong staged benchmark whose range scaling depends on source-head capacity,
  but still not a scalable/non-prescriptive recipe.
- Target propagation/local targets are now a ceiling and diagnostic, not the
  current scalable mainline. Exact `policy_reweighted_t1` is positive and
  survives retention, but full enumeration is not scalable. Simple
  approximation is paused after sparse/adaptive, replay-memory, corrected,
  online learned, pretrained learned, and sparse pairwise-preference variants
  failed scalability or Stage 1 stress. Continue only with new target
  construction or streaming validation.
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
  calc/exact, but heldout prompts only `0.0875`. Next work needs a mechanism
  for fresh-prompt generalization; a numeric amortized prior is the first
  promising diagnostic (`0.9125` heldout target accuracy), not yet a source gate.
- Lower-cost assignment is useful only when it changes scalability; uniform
  sampling, fixed refresh, and unique-uniform sampling are insufficient.
  Topk8+unique24 changes scorer slope to `O(C * 24)` and clears op19/op29
  staged gates. Corrected routed controls show two/four-hook cloned-output
  source and trusted handoff are causal, and routed execution is active-only.
  Shared output removes cloned-output parameter growth but hard-assignment
  handoff missed (`0.78`; matched `0.75`); semantic-distilled online hard
  memory is the first shared-output routed handoff family to replicate, pass
  op29, and train under stochastic minibatches with matched exposure. Heldout
  prompt failure means shared-output work now needs non-transductive memory or
  amortized/fresh-prompt credit, not more fixed-grid repeats.

## Paused Or Deprioritized Branches

These branches should not continue without a new mechanism:

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

1. Run the numeric amortized-prior heldout source gate; prompt-keyed memory alone fails heldout prompts.
2. Keep source objectives aimed at actual handoff/readout geometry,
   not one-metric recovery triggers or cheap selectors.
3. Do not tune forced-margin locally. Use automated recovery as the benchmark
   to beat; further forced-margin work must stress a new thesis-relevant axis
   such as many-calculator cost, cheaper assignment, or removal of hard
   assignment / true-result forcing. Do not rerun op19, shallow op29, op29
   low-LR recovery, completed op29 `rhead64` seeds, or the same op39 path.
4. If reducing hard-assignment cost, state the scalability hypothesis, compare
   against exact-grid, avoid more uniform count/fixed-refresh ladders, and use
   routing validation, op39 compute stress, or a changed estimator.
5. Use answer-derived result-boundary transfer as a bridge, not a recipe. Do
   not continue static critic/proposal/soft/regret-set target variants; use
   online/state-calibrated proposals, materially different target construction,
   or another credit-assignment family.

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
