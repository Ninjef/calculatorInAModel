# Phase 7 Ninth Task: Exact Result-Marginal Answer-Loss Gradient Gate

## Mission

Resolve the ambiguity left by the multi-sample result-space policy-gradient
negative before spending more long-run budget.

Phase 7 now knows all of this at once:

```text
1. Natural `0..19` result requests are representable and teachable.
2. Exact-grid upstream-open boundary-target teaching can learn hard result
   requests across seeds with semantic decoder movement exactly `0.0`.
3. Strict target-off retention is seed-fragile and did not robustly replicate.
4. Vanilla `K=16` result-space REINFORCE is wired, but its sampled gradient is
   anti-aligned with the boundary-target ceiling at initialization.
```

The next task should answer the missing question:

```text
Is the answer-loss gradient over the natural result-action distribution itself
aligned with the known good result direction, or was the sampled policy-gradient
gate dominated by finite-sample variance/control-variate weakness?
```

Do this by implementing an exact result-marginal expected answer-loss objective
over the `0..38` result action space and a Stage 0 gradient-agreement gate.

## Read First

Read these before editing or running:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-14-multisample-result-space-policy-gradient-gate.md
aiAgentProjectTasks/completed/phase7/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
```

For implementation, inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Why This Is The Next Best Task

Helpful knowledge from Phase 7:

- Exact full-grid coverage is worth keeping. It stabilized the upstream-open
  boundary-target branch where random resampling was only partial.
- The Phase 6 product decoder checkpoint is healthy for natural `0..19` result
  use. Oracle/readout checks are now regression checks only.
- Result-space action parameterization is the right abstraction for natural
  sum-only addition because the answer identifies a result, not a unique pair.
- The boundary-target branch is a useful supervised ceiling/control because it
  gives a known good direction without semantic decoder movement.
- The new REINFORCE plumbing is useful even though the Stage 0 gate failed: it
  provides a sampled score-function baseline and gradient diagnostic to compare
  against exact result-marginal gradients.

Less helpful as next work:

- More oracle/readout checks for natural `0..19`.
- More random-resampled or frozen-head boundary-target variants.
- More target-off retention reruns without a new mechanism.
- Canonical-query/protocol stabilization before robust retention or stronger
  discovery exists.
- Vanilla multi-sample result-space PG long runs while the Stage 0 cosine is
  negative.
- Jumping straight to actor-critic/NVIL/RELAX without first checking the exact
  expected result-action gradient. Learned baselines can reduce variance, but
  they cannot fix a fundamentally misaligned objective.

The fastest path to the end goal is to sort the fork:

```text
exact result-marginal gradient aligns -> sampled PG is the problem; improve
estimation or train with exact enumeration while the action space is small.

exact result-marginal gradient is also negative/near-zero -> answer-loss
expected-cost optimization is itself misaligned; pivot to surrogate gradients,
synthetic gradients, or stricter decoder-phase/bottleneck redesign.
```

## Claim Tested

Primary diagnostic claim:

```text
For natural `0..19` result-space actions, the exact expected answer-loss
gradient over result classes is positively aligned with the answer-derived
boundary-target ceiling on the exact grid.
```

Primary training claim, only if the diagnostic passes:

```text
An exact result-marginal expected answer-loss objective can train a hard
model-side calculator-result request without oracle operands, true result
targets, boundary-target CE/KL, sampled policy-gradient updates, direct operand
labels, or semantic decoder movement.
```

## Required Distinction From Earlier Work

Do not present this as a repeat of Phase 6 expected answer-loss.

Phase 6 expected answer-loss enumerated `20 x 20` independent operand pairs and
collapsed to wrong hard actions despite lowering expected cost. That negative
remains real, and broad independent-head repeats are not useful.

This task must instead use:

- `calculator_action_head=result_space`;
- result classes `0..38`, each mapped to the deterministic canonical valid
  calculator pair already used by result-space policies;
- exact-grid natural `0..19 x 0..19` batches;
- exact enumeration over `39` result actions, not sampled result actions;
- a gradient comparison against both the boundary-target ceiling and the
  sampled REINFORCE estimate.

## Training-Loss Guardrail

The exact result-marginal objective may use:

- answer NLL from forced result actions;
- the model's own result-action probabilities;
- exact enumeration over result classes;
- optional detached per-example cost centering or z-scoring;
- optional entropy over the model result policy.

It must not use these to construct the training loss:

- true sum labels;
- true operand labels;
- hard-best result CE;
- soft target CE/KL distilled from `softmax(-NLL/T)`;
- oracle operands;
- boundary-target gradients;
- sampled REINFORCE gradients;
- semantic decoder movement.

True sums, true operands, and hard-best parity may appear only in diagnostics.
The boundary-target gradient may be computed only for cosine comparison.

## Stage 0: Implementation And Exact Gradient Gate

Add a narrowly scoped result-space expected-loss path. Either:

```text
calculator_estimator=result_space_expected_answer_loss
```

or extend `calculator_estimator=full_enum_expected_answer_loss` so it supports
`calculator_action_head=result_space` without changing the existing
independent-head behavior.

Expected implementation shape:

1. Read `calculator_hook.result_proj` logits for the batch using the same
   result-space read path as the existing boundary-target and REINFORCE code.
2. Enumerate result classes `0..38` for each prompt.
3. Map each result class to the deterministic canonical calculator pair.
4. Force each candidate through the hard calculator/readout path and compute
   detached answer NLL costs:

```text
C_i(r) = answer NLL for prompt i when result class r is forced
```

5. Compute the model result policy:

```text
p_i(r) = softmax(result_logits_i / policy_temperature)[r]
```

6. Minimize exact expected answer NLL:

```text
L = mean_i sum_r p_i(r) * stopgrad(C_i(r))
```

7. Record metrics that make collapse and alignment visible:
   expected NLL, raw expected NLL, best NLL, true-result NLL for reporting only,
   learned-result NLL, expected-minus-best gap, learned-minus-best gap,
   learned-minus-true gap, result entropy/effective results, probability mass
   on best result, probability mass on true result for reporting only, hard
   learned-best fraction, and hard learned calculator-result accuracy.

Recommended CLI knobs:

```text
--expected-answer-loss-policy-temperature
--expected-answer-loss-cost-normalization none|center|zscore
--expected-answer-loss-entropy-weight
--expected-answer-loss-entropy-decay-steps
--expected-answer-loss-chunk-size
```

Add a diagnostic-only mode or extend the existing REINFORCE diagnostic so Stage
0 reports all three gradients on the same exact-grid batch:

- exact result-marginal expected-loss gradient;
- sampled result-space REINFORCE gradient using the existing `K=16` path;
- result-boundary hard-best CE gradient as the supervised ceiling/control.

Report gradient L2 and cosine similarities for:

- `calculator_hook.result_proj`;
- upstream trainable parameters;
- semantic decoder.

Stage 0 passes if:

```text
exact expected-loss result-proj grad L2 > 0
exact expected-loss upstream grad L2 > 0 when upstream is open
semantic decoder grad/delta L2 == 0.0
exact expected-loss vs boundary result-proj cosine > 0.0
exact expected-loss vs boundary upstream cosine > 0.0 when upstream is open
```

Also report:

```text
sampled PG vs exact expected-loss cosine
sampled PG vs boundary cosine
```

Interpretation:

- If exact expected-loss aligns but sampled PG does not, label it
  `result_space_pg_variance_or_control_variate_negative` and proceed to Stage
  1 exact-marginal training. Actor-critic/RELAX can remain future work, but
  exact enumeration is the faster controlled test while the action space is
  only `39`.
- If exact expected-loss is also negative or near-zero, stop. Do not train a
  long run. Label it
  `result_space_expected_answer_loss_alignment_negative` and recommend pivoting
  away from expected-cost/score-function objectives.

## Stage 1: Exact-Grid Result-Marginal Training

Run only if Stage 0 passes.

Use the fixed Phase 7 natural setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
calculator_result_vocab_size=39
calculator_action_head=result_space
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
answer_decoder_interaction=product
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
answer_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
input_proj_anchor_weight=0.0
reinforce_entropy_weight=0.0
exhaustive_grid_batch=true
```

Primary branch:

```text
policy_temperature=1.0
cost_normalization=none
entropy_weight=0.0
freeze_upstream_encoder=false
input_proj_lr=0.01
upstream_lr=0.0003
steps=800
snapshot_every=25
checkpoint_every=25
```

Allowed rescues, only if the primary branch has positive alignment and shows
meaningful movement but fails the hard-result gate:

1. `cost_normalization=center`
2. `cost_normalization=zscore`
3. `policy_temperature=0.5`

Run at most two rescues. Do not sweep broadly.

Stage 1 checkpoint selection must use learned-interface metrics:

- hard learned calculator-result accuracy;
- full-enum learned-result best fraction;
- mean learned-result minus best gap;
- expected answer-loss expected-minus-best gap;
- result entropy/effective results;
- canonical normal exact and calculator-result accuracy;
- injection-zero, forced-random, oracle-at-eval controls;
- semantic decoder movement exactly `0.0`;
- parameter deltas for result projection and upstream groups.

Stage 1 passes if the selected checkpoint reaches:

```text
hard learned calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
semantic decoder movement == 0.0
injection-zero and forced-random remain near chance
oracle-at-eval remains high
```

If Stage 1 does not pass after the allowed rescues, stop and write the negative
clearly. Do not run retention.

## Stage 2: Objective-Off Hard-Request Retention

Run only if Stage 1 passes.

Initialize from the selected Stage 1 checkpoint. Continue with hard result
requests and all discovery-specific objectives off:

```text
answer_loss_weight=1.0
expected_answer_loss_weight=0.0
result_boundary_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
input_proj_anchor_weight=0.0
reinforce_entropy_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=false
exhaustive_grid_batch=true
steps=400
snapshot_every=25
checkpoint_every=25
```

This is a retention/stability test, not the discovery claim. Select the best
post-start checkpoint by exact-grid hard learned calculator-result accuracy.

Stage 2 passes if:

```text
best post-start hard learned calculator-result accuracy >= 90% of Stage 1 selected accuracy
final hard learned calculator-result accuracy >= 0.70
semantic decoder movement == 0.0
all discovery-specific objective weights are exactly 0.0
injection-zero and forced-random remain near chance
oracle-at-eval remains high
```

## Required Outputs

Write a work-history entry under:

```text
aiAgentWorkHistory/phase7/
```

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
```

Record:

- code changes and validation commands;
- Stage 0 gradient table with exact expected-loss, sampled PG, and boundary
  gradient norms/cosines;
- Stage 1 branch table if run;
- Stage 2 retention table if run;
- final decision label.

Use one of these final labels:

```text
result_space_expected_answer_loss_alignment_negative
result_space_pg_variance_or_control_variate_negative
result_space_expected_answer_loss_stage1_negative
result_space_expected_answer_loss_retention_negative
result_space_expected_answer_loss_retained_positive
```

## Stop Rules

Stop before long training if Stage 0 exact expected-loss alignment is negative
or near zero.

Stop before Stage 2 if Stage 1 hard learned calculator-result accuracy is below
`0.70`.

Do not spend time on:

- oracle-only reruns;
- independent-head expected-answer-loss sweeps;
- vanilla result-space REINFORCE long runs;
- boundary-target retention reruns;
- canonical-query stabilization before this task clarifies the learning signal.

## Commit

After completing the task:

```bash
git status --short
git add src/model.py scripts/overfit_one_batch.py tests/test_model.py factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md aiAgentWorkHistory/phase7/<work-history-file>.md aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
git commit -m "Add result-marginal expected-loss phase 7 task"
git push
```
