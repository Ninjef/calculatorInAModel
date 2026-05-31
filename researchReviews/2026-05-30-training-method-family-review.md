# 2026-05-30 - Training method family review

## Why This Review

The project has accumulated many runs, and the user explicitly asked whether
the work is exploring high-leverage training methods or getting trapped in
small tweaks. This review groups runs by actual training-method family rather
than by seed, range, checkpoint, or diagnostic.

## Count

Strict count: ten distinct training-method families plus two
architecture/scaling extensions. Some of the ten are broad umbrellas with many
internal variants; count variants only when they change the learning signal or
training structure, not when they change seed, range, checkpoint, or a scalar
hyperparameter.

Training-method families tried:

1. Plain answer-loss / no-handoff discovery.
2. Vanilla result-space policy gradient and exact expected answer-loss
   gradients.
3. Decoder calibration as the main fix.
4. Direct feedback / output-projection boundary feedback.
5. Learned shadow-gradient feedback, including linear, online, refreshed,
   regularized, and Jacobian-featured variants.
6. Hard improvement assignment / full-grid local targets.
7. Approximate or sampled local targets, including uniform, unique, replay,
   adaptive, learned proposals, and pairwise preferences.
8. Policy-topk plus unique sampled assignment.
9. Staged bottleneck-to-additive transfer with freezing/anchoring/protection.
10. Source objectives aimed at additive handoff geometry, including
    forced-true, forced-margin, and recovery schedules.

Architecture/scaling extensions:

- Routed multi-hook / many-calculator training.
- Active-only routed execution and result-logit projection.

## What Should Stop

- Do not continue shadow-gradient variants that only change normalization,
  validation selection, dropout, feature scale, or simple loss shape.
- Do not continue uniform sampled-assignment ladders, fixed refresh ladders, or
  replay/local-target variants without a materially different estimator.
- Do not add more op19/op29 policy-topk seed replications as novelty.
- Do not treat source-selector proxy work as progress unless it replaces an
  actual trusted 600-step handoff gate across fresh families.

## What Deserves Compute

- Many-calculator scalability only when it changes real compute or parameter
  slope: active-only execution, shared/tied output projections, explicit
  routed compute accounting, or a routed training gate after such a change.
- Less-prescriptive credit assignment: answer-derived target construction,
  non-enumerative proposal mechanisms, or a training signal that avoids
  specifying the useful calculator result per prompt.
- Larger-range stress only with an explicit compute hypothesis, not as another
  full-grid forced-margin ladder.

## Strategic Update

The active-only routed execution patch is aligned because it removes a real
many-calculator waste term. But it should not absorb more local tweaking. After
shared/tied output projection or compute accounting, the project should return
to the central unsolved issue: scalable, less-prescriptive credit assignment
into the calculator-query policy.
