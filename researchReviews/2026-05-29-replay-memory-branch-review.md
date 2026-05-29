# 2026-05-29 Replay-Memory Branch Review

## Why This Review Exists

The replay-memory local-target branch produced the first strong sparse
approximation to exact `policy_reweighted_t1`, then accumulated enough stress
tests to decide whether the family should remain active.

This review exists to prevent future agents from rerunning nearby cache,
budget, reset, or streaming variants after the same failure mode has been
exposed from several angles.

## What Changed

The initial result was real:

- `memory_policy_reweighted_t1_u8_m24` beat raw uniform `u32` at 200 steps
  while scoring only 8 fresh result classes per step: `0.5900` exact calc /
  `0.5391` sampled normal versus `0.3350` / `0.3438`.
- It also survived an 800+200 retention gate: target `0.9600` calc /
  `0.9766` normal and retention `0.8600` calc / `0.8750` normal.
- A lower fresh-score point, `u2_m30`, improved the 200-step fixed-grid gate to
  `0.6025` calc / `0.6016` normal.

The stress tests changed the interpretation:

- `u2_m30` retained less strongly than `u8_m24` (`0.7850` calc / `0.7656`
  normal after retention).
- Top cached-candidate rescoring did not help: `u2_m30_r2` exactly tied
  no-rescore while costing twice as many forced scores, and heavier rescoring
  hurt.
- Reset windows exposed dependence on persistent prompt caches. At 199 steps,
  `reset100` reached only `0.4575` calc / `0.4453` normal and `reset50` only
  `0.2575` / `0.2812`, despite mostly restored target coverage.
- Streaming minibatches removed the strong lift. At 800 steps with batch `16`,
  exact `policy_reweighted_t1` and raw uniform `u32` both reached `0.2450`
  calc, `u8_m24` was merely comparable at `0.2650`, and `u2_m30` lagged at
  `0.1850`.

## Belief Changes

Replay memory is no longer a candidate scalable answer in its fixed
per-prompt-cache form.

The best explanation is that the fixed-grid positive is transductive: the
cache eventually observes most or all result classes for each prompt, then
reuses that prompt-specific table. That is useful evidence that cached
candidate proposals can matter, but it does not scale to many calculators,
larger models, or streaming/non-exhaustive prompt distributions.

The exact local-target ceiling remains strategically useful. It proves that a
non-backpropagated answer-derived target can train and retain calculator use.
But the approximation path must now change mechanism.

## What Should Stop

Stop treating these as novel:

- fixed replay-memory fresh-count ladders;
- cached-candidate rescore count ladders;
- reset interval or finite-cache tuning;
- streaming batch-size or longer-run checks of the same prompt-keyed cache;
- seed replications of fixed replay memory;
- small variations that still require the useful result to be present in a
  hand-coded candidate set.

These are now anti-rerun items, not active bets.

## What Deserves Compute

Local targets deserve more compute only if the proposal or estimator changes:

- learned/generalized candidate proposal, with a predeclared coverage and Stage
  1 lift gate against raw uniform `u32`;
- importance-corrected or bias-corrected sampled target;
- a target construction that can produce useful pressure when the correct
  result is absent from a small candidate set.

If the next local-target idea is not one of those, mainline compute should
return to source acquisition for additive handoff/readout geometry, where the
project already has a staged non-bottleneck success and a concrete gap: robust
source acquisition across fresh seeds without relying on brittle recovery
triggers or cheap selectors.

## Are We Closer To The Goal?

Yes, by closing a tempting but misleading scalability branch.

Replay memory briefly looked like a cheap approximation to exact local targets.
The stress tests show that the fixed-cache version does not solve scalable
credit assignment. That moves the project away from prompt-identity memory and
toward either learned proposal mechanisms or source objectives that directly
optimize transfer geometry.

## Steering Decision

Pause fixed replay-memory local-target proposals.

Keep exact `policy_reweighted_t1` as a ceiling and diagnostic. Continue
local-target approximation only with learned/generalized proposals, estimator
correction, or a different target construction. Otherwise, prioritize source
training objectives aimed at actual additive handoff/readout behavior.
