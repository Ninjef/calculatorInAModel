# 2026-05-30 Result-Boundary Static Approximation Steering Review

## Trigger

After the positive answer-derived result-boundary handoff, the project ran a
small cluster of static approximation tests: hidden/output critics, proposal
rescoring, adaptive proposal expansion, and soft target training. This review
checks whether this branch is still changing the project direction or becoming
a loop.

## What Changed

- The result-boundary source itself remains strategically valuable: it transfers
  causally into the trusted additive handoff without true-result forced-margin
  pressure.
- Direct amortized critics are too weak: best heldout argmin recovery was only
  `0.40` even with `24/39` train scores.
- Proposal rescoring is stronger but costly: step-800 top-16 proposals recover
  `0.96-1.00`, but they score `16/39` candidates; the ensemble version uses
  `32` train scores per prompt.
- Adaptive expansion has some signal but not enough leverage: margin-based
  expansion beats random and reaches `0.97` in the best ensemble setting, but
  fixed top-16 remains stronger and cheaper to reason about.
- Static soft-result targets do not help source training: matched step-200
  hard-best reached `0.5450` learned calc, while soft `t=1` reached `0.2900`
  and soft `t=4` reached `0.1350`.

## Stop

- Do not continue pointwise/pairwise/hybrid critic variants over the same
  hidden/output features.
- Do not continue static top-k count, LCB beta, ensemble-size, threshold, or
  expand-fraction sweeps as novelty.
- Do not continue static `soft_result` temperature ladders as the target
  construction answer.
- Do not wire these static approximations into source training and hope that
  dynamics solve the heldout/static gate.

## Continue

The answer-derived route is not dead, but the next method must change the
mechanism:

- validate proposal/target construction across evolving checkpoints, not just a
  fixed trained checkpoint;
- train a proposal model whose uncertainty is calibrated without ensemble costs
  close to enumeration;
- use a genuinely different target construction, such as set targets tied to
  uncertainty or regret, not just temperature-softened full-enum targets;
- or move to a different less-prescriptive credit-assignment family.

## Decision

```text
static_result_boundary_approximation_paused
```

Result-boundary remains a useful answer-derived bridge and benchmark, but
static approximations have reached a local boundary. Future compute should
either test evolving-state/generalization or change target construction
materially; otherwise, move to a different credit-assignment mechanism.
