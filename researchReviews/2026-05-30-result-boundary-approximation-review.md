# 2026-05-30 Result-Boundary Approximation Review

## Trigger

The answer-derived result-boundary source now has a trusted additive handoff
positive, but the immediately tempting path is to keep trying small critic
variants to replace full forced-result enumeration. This review decides whether
that remains a good compute target.

## What Changed

- The May 13 full-grid result-boundary source checkpoint transfers causally
  into the additive frozen-policy gate: `0.8825` final eval, `0.8425` step-600
  normal, zero-injection `0.0000`, forced-random `0.0391`, learned calc
  `0.9922`.
- This shows true-result forced-margin pressure is not strictly required for
  staged transfer.
- A pointwise hidden/output amortized critic failed to approximate the
  full-enum target on heldout prompts:
  - `k=8`: heldout argmin recovery `0.0800`, `0.0800`, `0.1700` at steps
    `0`, `100`, `800`;
  - `k=24`: `0.2600` at step `0`, `0.1900` at step `800`.
- Rank-aware training helped but did not clear a useful gate:
  - pairwise `k=24`: `0.2600` at step `0`, `0.4000` at step `800`;
  - hybrid `k=24`: `0.2000` at step `0`, `0.2700` at step `800`.

## Stop

- Do not continue pointwise, pairwise, hybrid, hidden-size, epoch-count, or
  learning-rate variants of the same hidden-state plus candidate-output-vector
  critic as novelty.
- Do not wire this critic into source training and hope training dynamics will
  fix the heldout argmin problem.
- Do not count top-5 recovery as sufficient: the source objective needs a
  confident target or a principled uncertainty mechanism, not a broad candidate
  hint that still requires near-enumeration.

## Continue

Result-boundary target construction remains strategically interesting because
it is answer-derived and transferred causally. Continue only if the next method
changes one of these things:

- target construction: train on a set/interval/soft target that tolerates
  uncertainty instead of requiring exact argmin prediction;
- estimator: use uncertainty-aware sampling that expands compute only where the
  critic cannot separate candidates;
- generalization mechanism: validate across evolving model checkpoints or
  prompt ranges before source training, not only fixed-grid interpolation.

## Decision

```text
hidden_output_boundary_critic_family_paused
```

The project is closer to the overarching goal only in the negative sense: the
answer-derived route is still promising, but simple amortized critics are not
the scalable bridge. Future work should change target construction or
uncertainty handling rather than polish this critic family.
