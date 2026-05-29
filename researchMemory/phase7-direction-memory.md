# Phase 7 Direction Memory

Status: active synthesis
Last updated: 2026-05-29

This file consolidates Phase 7 lessons by research direction. It is meant to
be easier to retrieve than the chronological fact sheet.

## Central Lesson

Natural `0..19` result-level calculator use is not blocked by architecture or
downstream decoding. It is blocked by scalable credit assignment into the
calculator-query policy.

The strongest positives use scaffolding, assignment, or staged transfer. The
missing result is non-prescriptive from-scratch discovery.

## Direction: Oracle And Wiring Checks

Status: paused

Memory:

- Correct calculator outputs or oracle queries let downstream answer layers
  solve the task.
- This is a wiring/control result only.
- Do not present oracle success, oracle-at-eval recovery, injection-zero, or
  forced-random checks as strategic progress except when validating new wiring.

Representative evidence:

- `CLAUDE.md` guardrails
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`

## Direction: Generic Retention After Teaching

Status: paused

Memory:

- Earlier phases established that identifiable/scaffolded protocols can often
  be retained after direct supervision or local targets are removed.
- Phase 7 target-off work is only strategic when it tests a new interface or a
  real stability question.

Representative evidence:

- `RESEARCH_STATE.md`
- `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`

## Direction: Plain Answer-Loss Discovery

Status: paused without a new mechanism

Memory:

- Vanilla result-space REINFORCE failed because its gradient aligned with the
  raw expected-cost gradient, and that gradient anti-aligned with the boundary
  ceiling.
- Exact result-marginal answer loss did not fix the direction.
- Decoder calibration improved local signs but still collapsed in Stage 1.
- Conclusion: the failure is not mainly finite-sample variance or a slightly
  weak decoder.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Direct Feedback And Shadow Gradients

Status: paused without a new dynamics mechanism

Memory:

- Output-projection direct feedback passed a local Stage 0 alignment gate but
  failed Stage 1 discovery.
- Fixed fit-once linear shadow feedback produced excellent same-batch alignment
  but failed early lift and failed heldout generalization.
- Many simple online MLP shadow variants improved heldout alignment metrics but
  did not create useful training dynamics.
- Stage 0 gradient alignment alone is not a sufficient go signal.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Hard Improvement Assignment

Status: active but constrained

Memory:

- Hard answer-loss improvement assignment is the strongest bottleneck source
  training signal so far.
- Always-on assignment can train natural result-level calculator policies.
- Plain target-off decay failed, and the method is expensive/prescriptive
  because it scores candidate calculator results.
- The open question is scalability: can this be approximated or replaced
  without losing the source-policy result?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-convergence-gate.md`
- `HYPOTHESIS_LEDGER.md`

## Direction: Non-Bottleneck Direct Training

Status: paused unless adding a new causal-use mechanism

Memory:

- Direct additive non-bottleneck training with answer loss plus hard assignment
  learned mostly through the neuron path; calculator-result accuracy stayed
  near chance.
- Simple zero-injection causal-gap pressure made bypass worse but did not make
  the calculator query correct.
- Non-bottleneck success currently comes from staged transfer, not direct
  discovery.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-hard-assignment-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-causal-gap-gate.md`

## Direction: Bottleneck-To-Additive Handoff

Status: active but constrained

Memory:

- A bottleneck-trained calculator policy can be loaded into an additive
  non-bottleneck model and used causally if the policy is frozen/protected.
- Unprotected compatible transfer destroys the policy quickly.
- Source quality and representation geometry matter; source accuracy alone is
  not enough.
- This proves non-bottleneck viability but is not yet scalable or
  non-prescriptive.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-src6-selected-continuation-readout.md`

## Direction: Source Checkpoint Selection

Status: paused for cheap proxies; actual handoff gates remain trusted

Memory:

- Source normal/calculator accuracy is not a reliable selector.
- Actual 600-step frozen-policy additive handoff is the most reliable tested
  source-checkpoint gate for fresh families.
- Several cheaper proxies failed or were mixed: frozen-state readout,
  forced-result geometry, 25/50/100-step loss slope, ridge over early handoff
  traces, 500-step standalone selection on fresh `src6`, and 500-step embedded
  probe normal score on fresh seed `11`.
- Embedded probes are useful logging/triage, but 500-step normal alone is not
  validated as a selector.
- Selector-cost reduction is no longer the default frontier. New selector work
  must either beat the 600-step gate on fresh families or change source
  training pressure directly.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`
- `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`
- `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`

## Direction: Source Acquisition For Transfer Geometry

Status: active

Memory:

- Stabilized source acquisition can produce transferable policies in some
  seeds, but high source performance does not guarantee transfer.
- Seed-9-style no-decay stabilization plus continuation/readout can clear the
  non-bottleneck gate; seed-10-style runs show hostile geometry can persist.
- The source objective can keep improving learned calculator accuracy while
  worsening additive handoff geometry; seed 10 is the clearest warning.
- A first direct geometry objective is mixed-positive: a forced-true additive
  readout auxiliary during bottleneck source acquisition made the true result
  the best forced additive result on `59%` of a small `0..9` grid versus `0%`
  baseline, but weakened source policy accuracy at the same budget.
- Scheduling fixed the first tradeoff in the same small gate: delaying the
  forced-true additive auxiliary to step `50` improved source calc/final eval
  over baseline (`0.3900`/`0.4000` vs `0.3500`/`0.3800`) while retaining
  strong additive geometry (`forced_best_true=0.5100` vs baseline `0.0000`).
- The first full-grid scheduled gate is positive: at matched 200-step
  `operand_max=19` source checkpoints, scheduled aux nearly tied source
  accuracy but improved forced-result geometry (`forced_best_true=0.2125` vs
  `0.0000`) and standalone 600-step handoff (`0.4150` final eval vs `0.2525`).
- Longer scheduled source training compounded through source step `600`: the
  step-600 checkpoint reached `forced_best_true=0.9800` and standalone
  600-step handoff final eval `0.7725`. Step `800` had perfect forced-result
  geometry but worse handoff (`0.6750`), so final source checkpoint is not
  automatically best.
- The promising next direction is continuation/readout from the scheduled
  step-600 handoff lineage. Keep standalone 600-step handoff verification in
  the loop; forced-result geometry alone remains a triage signal.

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-floor.md`
- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-readout.md`
- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-seed10-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`
- `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`
- `researchReviews/2026-05-29-scheduled-source-geometry-review.md`

## Direction: Target Propagation / Local Targets

Status: active candidate with Stage 1 retention partial positive

Memory:

- This direction has not yet received the same empirical treatment as the
  answer-loss and shadow-gradient families.
- It is strategically interesting because it changes the credit-assignment
  family rather than tuning another proxy.
- The first exact-grid Stage 0 gate is partially positive: sharp
  current-policy-reweighted targets and weakly proximal logit-descent targets
  produce gradients aligned with the hard boundary ceiling, while expected
  answer loss remains anti-aligned.
- A 200-step Stage 1 lift gate showed `policy_reweighted_t1` reaches `0.5600`
  exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy,
  beating the failed expected-loss baseline and slightly beating the
  same-budget hard-boundary run.
- An 800-step target-training plus 200-step answer-only retention gate showed
  `policy_reweighted_t1` is nonmonotonic during target training but can finish
  retention strongest: `0.8925` exact-grid calculator-result accuracy and
  `0.8750` sampled normal, with injection-zero and forced-random controls low.
- A first sparse approximation gate was mixed-negative: naive top-k/uniform
  sampled candidate sets underperformed badly unless they scored nearly the
  full result vocabulary (`u32` only `0.3350` exact-grid calc at 200 steps,
  `u36` `0.4100`, while full-vocabulary `u39` reached `0.6250`).
- A simple adaptive loss-neighborhood proposal was negative: at similar scoring
  budgets it underperformed raw uniform `u32` because expansion clustered into
  fewer unique candidates and lower true-result coverage.
- The result is not yet a scalable or non-prescriptive method. The tested
  targets still need broad forced-result scoring.
- Next work should prioritize a learned candidate proposal or bias/variance
  correction for `policy_reweighted_t1`; otherwise pivot back to source
  acquisition for handoff geometry. Do not rerun raw uniform/top-k ladders or
  simple low-loss-neighborhood expansion as novelty.

Representative evidence:

- `SOLUTION_IDEAS.md`
- `RESEARCH_STATE.md`
- `researchReviews/2026-05-29-phase7-local-target-approximation-review.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`
