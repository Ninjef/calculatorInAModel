# 2026-06-02 - Route-Excluded Shared-Prior Preflight Review

## Trigger

The project is trying to break the per-calculator prompt-memory/prior-scaling
problem. The route-heldout diagnostic was promising enough to justify an
actual source-training gate, but a follow-up audit found the first route
exclusion plumbing landed on the wrong training branch.

## What Changed

The prompt-keyed training loop now actually passes
`memory_update_exclude_routes` into
`result_boundary_prompt_hard_memory_loss(...)`. A smoke run verified that
score-eligible and update-excluded fractions move in the training curve.

The corrected op9 preflight did not pass: final/source accuracy was `0.510`,
heldout prompt accuracy was `0.050`, prior heldout accuracy was `0.050`, and
the excluded route remained near chance. No handoff was run.

## What Should Stop

- Do not use the first route-excluded op9 preflight as evidence; it was run
  before the prompt-memory plumbing fix.
- Do not rerun short op9 preflights or route-heldout diagnostics as if they
  answer the full shared-prior source question.
- Do not drift back into cap-value, seed, validation-threshold, or prior-fit
  cadence ladders unless the change reduces per-calculator target discovery or
  candidate scoring.

## What Deserves Compute

1. A full op19 route-excluded source gate using the strongest known capped or
   full-memory numeric-prior dynamics, followed by trusted frozen-policy
   handoff only if source heldout and excluded-route quality pass.
2. A genuinely shared/global target-prior mechanism that pools target learning
   across routed calculators during training, not only in post-hoc diagnosis.
3. A non-enumerative credit-assignment mechanism that removes answer-derived
   candidate scoring, if available.

## Strategic Status

This branch remains strategically aligned because it attacks the many-calculator
scaling failure directly. The preflight is not a reason to abandon shared
priors; it is a warning that the lightweight op9 setup was too weak and that
future agents should either spend the full source-gate budget or change the
mechanism more substantially.
