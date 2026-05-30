# Research State

Last updated: 2026-05-30

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
  failed scalability or Stage 1 stress. Continue only with a materially
  different estimator/target construction or explicit streaming validation.
- Less-prescriptive answer-derived target construction. The old full-grid
  result-boundary target source transfers causally into the trusted additive
  frozen-policy gate (`0.8825` final, `0.8425` step-600 normal, zero-injection
  `0.0000`, learned calc `0.9922`). This shows true-result forced-margin is
  not strictly required for staged transfer, but full forced-result enumeration
  and frozen transfer remain unsolved scalability/prescriptiveness issues.
  Hidden-state/candidate-output critics do not look like the scalable bridge:
  pointwise recovery stayed at `0.08-0.26`, and the best pairwise result was
  only `0.40` heldout argmin recovery while using `24/39` result scores.
- Lower-cost assignment is useful only when it changes scalability. Uniform
  sampled hard assignment is negative: sample16/sample32 on op19 `rhead64`
  failed to preserve exact source signal (`0.3650`/`0.4050` best snapshots vs
  `0.8625` exact). Fixed exact-target refresh is also mixed-negative:
  refresh2/refresh5 reached only `0.5875`/`0.4950` best snapshots. Next
  attempts need adaptive freshness, coverage-aware/structured proposals, or a
  non-enumerative credit signal.

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
- Slight weight/seed/length changes to already failed source-stabilization
  recipes unless tied to a new transfer-geometry objective.

## Next 1-3 Experiments

1. Continue local targets only via materially different target construction,
   estimator changes, or explicit streaming/generalization validation.
2. Keep source objectives aimed at actual handoff/readout geometry,
   not one-metric recovery triggers or cheap selectors.
3. Do not tune forced-margin locally. Use automated recovery as the benchmark
   to beat; further forced-margin work must stress a new thesis-relevant axis
   such as many-calculator cost, cheaper assignment, or removal of hard
   assignment / true-result forcing. Do not rerun op19, shallow op29, op29
   low-LR recovery, completed op29 `rhead64` seeds, or the same op39 path.
4. If reducing hard-assignment cost, state the scalability hypothesis up front
   and compare against the exact-grid ceiling. Do not run more uniform sampled
   count ladders or fixed refresh-interval ladders on op19 `rhead64`; improve
   freshness/coverage/target quality or change the estimator.
5. Use answer-derived result-boundary transfer as a bridge, not a recipe:
   next work should approximate or replace the full forced-result enumeration
   that selected the best-result target. Do not continue pointwise/rank
   hidden-output critic variants; use different target construction,
   uncertainty-aware compute, or validation across evolving model states.

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
