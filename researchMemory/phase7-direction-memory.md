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
- Continuation/readout from the scheduled step-600 handoff lineage was
  mixed-positive but below gate: continuation reached `0.7775`, 600-step
  readout reached `0.8175`, and extended readout reached `0.8475`, with low
  controls but learned calc stuck around `0.5391`.
- The source-policy diagnosis was confirmed: a gentle low-LR recovery phase
  from the scheduled step-600 checkpoint (`lr=3e-4`, forced-true weight `0.1`,
  30 steps) raised source calc from `0.5800` to `0.7950` while preserving
  low forced-true loss, improved 600-step handoff to `0.8425`, and cleared
  continuation/readout at `0.9320` with low zero/random controls.
- The recovery effect replicated on fresh seed `14`: scheduled step-600 source
  eval `0.6675` recovered to `0.8850`, and the recovered checkpoint reached
  `0.9600` final eval under the trusted 600-step frozen-policy handoff, with
  learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`
  at the final snapshot.
- The late-source recovery phase can be automated in a single source run:
  a fixed step-600 switch to LR multiplier `0.1` and forced-true weight `0.1`
  reached source eval `0.8775` and a trusted 600-step handoff final eval
  `0.9400`, with learned calc `0.8725`, injection-zero `0.0800`, and
  forced-random `0.0775` at the final handoff snapshot.
- A simple adaptive source-accuracy trigger is mixed-positive: after the
  adaptive recovery switch was wired to lower both LR and forced-true weight,
  `result_policy_argmax_result_accuracy >= 0.65` fired at step `528` and the
  trusted 600-step handoff reached `0.9850` final eval, exceeding the fixed
  step-600 switch. The tradeoff is higher controls (`0.1325` zero and
  forced-random) and lower source final eval (`0.8250`), so it needs fresh-seed
  replication or a smoother/conjunctive trigger before treating it as a
  validated selector.
- The fresh-seed replication was negative for raw source-accuracy thresholding:
  on seed `17`, the same `>=0.65` trigger never fired, source final eval was
  `0.6100`, and trusted 600-step handoff reached only `0.6825`. A matched fixed
  step-600 control did better (`0.7450` source, `0.7675` handoff) but still
  missed the high gate. Do not treat raw argmax source accuracy as a validated
  adaptive transition criterion.
- Forced-true loss is a better one-metric trigger on the hard seed 17:
  `additive_forced_true_loss <= 0.05` fired at step `500`, improved source final
  to `0.7225`, and reached `0.7625` trusted 600-step handoff with low controls.
  This mostly matched the fixed step-600 control (`0.7675`) and beat the
  no-trigger source-accuracy branch (`0.6825`), but still missed the high gate.
- Adding EMA/patience to the forced-loss trigger improved timing on the same
  hard seed: beta `0.8`, patience `10`, min step `500` fired at step `509`,
  raised source final to `0.7625`, and improved trusted 600-step handoff to
  `0.8025` with low forced-random (`0.0325`) but still below the high gate.
- A hard conjunctive source-accuracy gate was too conservative on the same
  hard seed: forced-loss readiness was satisfied, but requiring
  `result_policy_argmax_result_accuracy >= 0.70` never fired, source final
  stayed `0.6100`, and handoff returned to `0.6825`.
- A new contrastive source-geometry objective is mixed-positive in the small
  gate: scheduled additive forced-margin training reached `0.4100` source calc
  / `0.3800` final eval, `forced_best_true=0.6200`, and `top3=0.7500`, but its
  50-step slope final loss (`1.0238`) was worse than scheduled forced-true
  (`0.7979`).
- Forced-result geometry alone remains a triage signal; actual handoff/readout
  gates remain decisive.

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
- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`
- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-automated-scheduled-source-recovery.md`
- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`
- `aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md`
- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md`
- `researchReviews/2026-05-29-scheduled-source-geometry-review.md`

## Direction: Target Propagation / Local Targets

Status: active only with learned/corrected proposals or new target construction

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
- A replay-memory proposal was the first positive sparse approximation: scoring
  `8` fresh uniform results per step and reusing cached low-loss candidates
  beat raw uniform `u32` at 200 steps (`0.5900` vs `0.3350` exact-grid calc)
  and reached `0.8600` calc / `0.8750` sampled normal after an 800+200
  answer-only retention gate.
- Lowering the fresh scoring budget improved the short gate through `u2_m30`:
  at 200 steps `u2_m30` reached `0.6025` exact-grid calc and `0.6016` sampled
  normal, while `u1_m31` weakened to `0.4075`/`0.4219`. The long `u2_m30`
  retention gate was positive but weaker than `u8_m24`: `0.7850` calc /
  `0.7656` normal after answer-only retention.
- Simple cached-candidate rescoring did not improve that retention weakness:
  `u2_m30_r2` tied no-rescore in both 200-step and 800+200 gates, while
  heavier `r4/r8` rescoring hurt short-gate learning.
- Finite reset windows exposed a transductive dependency: at 199 steps,
  no-reset `u2_m30` reached `0.5925` exact calc / `0.5938` sampled normal,
  while `reset100` reached only `0.4575` / `0.4453` and `reset50` only
  `0.2575` / `0.2812`, even though target true-candidate coverage mostly
  recovered between resets.
- Streaming minibatches with prompt-keyed caches removed the strong replay
  lift. At 800 steps with batch `16`, exact `policy_reweighted_t1` and raw
  uniform `u32` both reached `0.2450` exact calc, `u8_m24` was only comparable
  at `0.2650`, and `u2_m30` lagged at `0.1850`.
- The durable lesson is negative for fixed hand-coded proposals. Exact
  `policy_reweighted_t1` remains a useful ceiling, but fixed replay memory is
  not a scalable method. Next local-target work needs learned/generalized
  proposal memory, estimator correction, or a different target construction.

Representative evidence:

- `SOLUTION_IDEAS.md`
- `RESEARCH_STATE.md`
- `researchReviews/2026-05-29-phase7-local-target-approximation-review.md`
- `researchReviews/2026-05-29-replay-memory-branch-review.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md`
- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md`
