# Phase 7 Eighth Task: Multi-Sample Result-Space Policy-Gradient Gate

## Mission

Make the next Phase 7 move a genuine estimator-family test, not another
boundary-target or retention rerun.

The current Phase 7 state is:

```text
Exact-grid upstream-open result-boundary teaching can learn natural result
requests across seeds, but target-off retention is seed-fragile and does not
robustly clear the strict replication gate.
```

The next best bet is therefore:

```text
Can multi-sample score-function training over natural result requests discover
a hard calculator-result protocol from answer loss, using per-prompt or
leave-one-out baselines to reduce variance?
```

This is deliberately different from the boundary-target branch. The
boundary-target recipe enumerates the frozen decoder and trains a supervised
result target. This task should use the sampled calculator action's realized
answer loss as the policy-gradient learning signal.

## Read First

Read these before editing or running:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_1_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-13-exact-grid-retained-positive-seed-replication.md
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

- Exact full-grid coverage matters. Random-resampled `batch_size=400` was
  partial; exhaustive `20 x 20` batches stabilized the upstream-open boundary
  branch.
- The natural product decoder and full-enum result landscape are healthy.
  Oracle/readout and true-result best gates are infrastructure now, not
  discoveries.
- Upstream-open result-space requests can be taught without semantic decoder
  movement.
- Dense exact-grid snapshots and full-enum learned-result diagnostics are the
  right way to select checkpoints.
- The boundary-target branch is now a useful supervised ceiling/control for
  estimator comparisons.

Less helpful as next work:

- More oracle/readout checks, unless code changes could have broken wiring.
- More frozen linear/MLP result-head or random-resampled boundary-target runs.
- More target-off retention reruns that do not introduce a new learning signal
  or diagnose seed fragility.
- Moving directly to canonical-query/protocol stabilization as if exact-grid
  retention had robustly replicated.
- Repeating Phase 1 single-sample REINFORCE. Phase 1 already showed that a
  single-sample independent-operand estimator with a moving scalar baseline did
  not discover the intended protocol.

This task is not that Phase 1 repeat. It uses the Phase 7 natural result-space
action, exact-grid batches, multi-sample per-prompt advantages, and gradient
agreement diagnostics against the known boundary-target ceiling.

## Claim Tested

Primary claim:

```text
In natural 0..19 addition, a multi-sample result-space score-function
estimator can train the model-side calculator request toward the answer-loss
preferred result class without direct operand labels, oracle operands,
boundary-target CE/KL, deterministic Concrete relaxation, or semantic decoder
movement.
```

Secondary claim, only if Stage 1 passes:

```text
The learned hard result request survives answer-only target-off continuation
with all policy-gradient/discovery-specific objectives disabled.
```

## Required Distinction From Earlier Work

Do not present this as the first REINFORCE attempt. It is not.

The task must explicitly compare against Phase 1:

- Phase 1 used independent A/B operand sampling.
- Phase 1 used a single sample per prompt.
- Phase 1 used a moving scalar baseline.
- Phase 1 optimized true-operand-style protocols in an earlier, less stable
  bottleneck/diagnostic regime.

This task must use:

- `calculator_action_head=result_space`;
- result actions `0..38`, mapped to deterministic valid calculator calls;
- exact-grid `0..19 x 0..19` batches for the main gate;
- at least one multi-sample baseline: per-prompt mean or leave-one-out;
- learned-result/full-enum/counterfactual diagnostics, not answer accuracy
  alone.

## Stage 0: Implementation And Gradient Estimator Gate

Add the smallest implementation needed for result-space policy-gradient
training.

Expected implementation shape:

1. Allow `calculator_action_head=result_space` with
   `calculator_estimator=reinforce`, or add a narrowly named equivalent if that
   is cleaner.
2. In result-space policy-gradient mode, sample a result class from
   `calculator_hook.result_proj` logits, map it to the existing deterministic
   canonical pair, and run the real hard calculator path.
3. Record trace fields that make the sampled action auditable:
   `result_pred`, `result_logp`, `sampled_logp`, `result_entropy`,
   `a_pred`, and `b_pred`.
4. Make `sampled_logp` equal the sampled result log-probability for
   result-space policies, not the sum of synthetic A/B canonical log-probs.
5. Add a multi-sample policy-gradient path in `scripts/overfit_one_batch.py`.
   It may duplicate the exact-grid batch `K` times or loop over `K` stochastic
   passes, but it must compute per-prompt advantages.
6. Implement at least:
   - `reinforce_baseline_mode=global_ema` for backward compatibility;
   - `reinforce_baseline_mode=per_prompt_mean`;
   - `reinforce_baseline_mode=leave_one_out`.
7. Preserve the old single-sample independent-operand behavior unless the task
   can show all existing tests and CLI validation still cover it.

Before any real training, run a fixed-batch estimator diagnostic:

- exact exhaustive grid, `400` prompts;
- `K >= 8` samples per prompt, preferably `K=16` if runtime is reasonable;
- semantic decoder frozen;
- no boundary-target loss;
- no oracle operands;
- report mean/standard deviation of advantages, policy-gradient objective,
  result entropy, sampled result accuracy, and gradient norm into
  `calculator_hook.result_proj` and upstream groups.

Add a gradient-agreement diagnostic against the boundary-target ceiling:

- Compute the existing hard-best result-boundary CE gradient on the same exact
  grid.
- Compute the multi-sample policy-gradient estimate on the same exact grid.
- Report cosine similarity and relative norm for `calculator_hook.result_proj`
  and, when upstream is open, for upstream trainable parameters.
- This is a diagnostic only. The policy-gradient branch must not use the
  boundary-target gradient for updates.

Stage 0 passes if:

```text
result-proj PG gradient L2 > 0
semantic decoder gradient/delta L2 == 0.0
PG-vs-boundary result-proj gradient cosine > 0.0
per-prompt or leave-one-out baseline has lower gradient/advantage variance than global EMA
```

If the cosine is negative or near zero, stop and report that vanilla
multi-sample result-space policy gradient is not currently aligned with the
known answer-derived target. Do not train a long run by force.

## Stage 1: Exact-Grid Result-Space Policy-Gradient Training

Run only if Stage 0 passes.

Use the fixed Phase 7 natural setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
calculator_result_vocab_size=39
calculator_action_head=result_space
calculator_estimator=reinforce
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
answer_decoder_interaction=product
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
result_boundary_target_loss_weight=0.0
input_proj_anchor_weight=0.0
exhaustive_grid_batch=true
```

Primary branch:

```text
reinforce_baseline_mode=leave_one_out
reinforce_num_samples_per_prompt=16
answer_loss_weight=0.0
freeze_upstream_encoder=false
input_proj_lr=0.01
upstream_lr=0.0003
steps=800
snapshot_every=25
checkpoint_every=25
```

If runtime is too high, reduce to `K=8` before reducing the exact-grid
discipline. Keep the exhaustive grid unless there is a clear memory/runtime
blocker.

Allowed single rescue:

- switch `leave_one_out` to `per_prompt_mean`, or
- lower `input_proj_lr` by `3x`.

Do not do both unless the first branch was a near-pass and the work history
justifies it.

Stage 1 selection should use exact-grid learned-interface metrics, not final
answer exact alone:

- hard learned calculator-result accuracy;
- full-enum learned-result best fraction;
- learned-result minus best-result gap;
- result entropy and confidence;
- canonical normal exact and calculator-result accuracy;
- injection-zero, forced-random, oracle-at-eval controls;
- semantic decoder movement exactly `0.0`;
- parameter deltas for result projection and upstream groups.

Stage 1 passes if the selected checkpoint reaches:

```text
hard learned calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
injection-zero near chance
forced-random near chance
oracle-at-eval near 1.0
semantic decoder movement exactly 0.0
```

If Stage 1 fails below `0.30`, stop. That would be a meaningful negative for
this estimator in its cleanest result-space form.

If Stage 1 lands in `0.30..0.70`, do not immediately run schedule sweeps.
First compare gradient variance, entropy collapse, and PG-vs-boundary cosine
over training snapshots to decide whether the failure is estimator variance,
premature collapse, or feature movement instability.

## Stage 2: Objective-Off Retention

Run only if Stage 1 passes.

Initialize from the selected Stage 1 checkpoint and turn off all
policy-gradient/discovery-specific objectives:

```text
calculator_estimator=gumbel_concrete_interface
calculator_action_head=result_space
relaxed_calculator_temperature=1.0
relaxed_calculator_final_temperature=1.0
relaxed_calculator_hard_forward=true
relaxed_calculator_entropy_weight=0.0
answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
exhaustive_grid_batch=true
steps=400
snapshot_every=25
checkpoint_every=25
```

Rationale: Stage 1 policy gradient is the discovery signal. Stage 2 should ask
whether the hard result request is usable by the ordinary answer-loss
continuation path after the estimator is removed.

Stage 2 passes if:

```text
best post-start hard result accuracy retains >= 90% of selected Stage 1 hard result accuracy
final hard result accuracy >= 0.70
final full-enum learned-result best fraction >= 0.70
semantic decoder movement exactly 0.0
all discovery-specific objective weights exactly 0.0
```

This retention is allowed because it tests a genuinely new discovery signal.
Do not describe it as a new discovery that target-off retention can happen in
general.

## Optional Stage 3: Joint-Pair Policy-Gradient Stretch

Only run this if result-space Stage 1 passes or near-passes.

Implementing joint-pair policy-gradient first is tempting, but it expands the
action space from `39` result actions to `400` pair actions and reintroduces
same-result underidentification. Result-space policy gradient is the cheap,
high-signal estimator gate.

If the result-space branch is positive, open a narrow joint-pair PG stretch:

- `calculator_action_head=joint_pair`;
- sample pairs from the joint `20 x 20` policy;
- score by downstream answer loss under the real calculator result;
- use per-prompt or leave-one-out baselines;
- judge by result-equivalent accuracy and learned calculator-result accuracy,
  not true pair exact.

Stop after a smoke/gradient gate unless the result-space branch passed cleanly.

## Reporting Requirements

Use run root:

```text
runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate
```

On completion, update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-14-multisample-result-space-policy-gradient-gate.md
```

The work history must include:

- exact claim tested;
- code changes;
- validation commands and results;
- Stage 0 estimator diagnostics, including PG-vs-boundary gradient cosine;
- exact command lines for every run;
- run paths;
- selected checkpoint paths;
- fast-gate table;
- full diagnostic table;
- final objective weights;
- parameter movement summary;
- comparison to Phase 1 single-sample REINFORCE and Phase 7 boundary-target
  ceiling;
- go/no-go recommendation.

When complete, move this task to:

```text
aiAgentProjectTasks/completed/phase7/
```

then commit and push.

## Decision After This Task

If multi-sample result-space PG passes Stage 1 and Stage 2:

```text
Phase 7 has a stronger natural result-level discovery result than the
boundary-target branch because the discovery signal is sampled answer loss,
not an explicit boundary-target teacher. Next move should be joint-pair PG or
canonical-query stabilization.
```

If Stage 1 passes but Stage 2 fails:

```text
The estimator can discover natural result requests, but hard answer-loss
continuation is unstable. Diagnose retention fragility instead of rerunning
teaching.
```

If Stage 0 aligns but Stage 1 fails:

```text
The gradient estimate is directionally plausible but the optimization remains
too noisy or unstable. Consider actor-critic/NVIL-style learned baselines or
direct feedback alignment.
```

If Stage 0 does not align:

```text
Do not spend long-run budget on vanilla result-space policy gradient. Move to
surrogate/shadow-calculator gradients, synthetic gradients/direct feedback
alignment, or a stricter decoder-phase bottleneck.
```
