# Phase 7 Overarching Plan: Natural Result-Level Interface Discovery

## Mission

Phase 7 should move the project from identifiable calculator-query discovery
toward natural arithmetic calculator use.

Phase 6 established the strongest current learned-interface result:

```text
In an identifiable `sum_left_operand` setting, answer-derived deterministic
hard-forward / soft-backward Concrete training can discover a hard calculator
protocol, and answer-only continuation can retain that protocol after every
local, auxiliary, expected-loss, relaxed, and anchor objective is exactly off.
```

Phase 6 also established the next blocker:

```text
In natural sum-only addition, the answer identifies the result but not a unique
operand pair. The correct result group gets essentially all answer-derived
target mass, but that mass is spread across many same-sum pairs. Independent
operand heads therefore have no unique pair target to discover.
```

Phase 7 should therefore ask:

```text
Can a model-side interface learn to make causally useful calculator calls for
natural `0..19` addition when the action parameterization matches the
result-level information present in the answer loss?
```

This is not a retreat from calculator use. It is the natural next step after
Phase 6: stop asking natural answer loss to identify an arbitrary true operand
pair, and instead require the learned interface to produce calculator results
that answer the task through the calculator path.

## Read First

Before doing Phase 7 work, read these files in this order:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/completed/phase6/2026-05-10-phase-6-overarching_plan-Identifiable-local-interface-discovery.md
aiAgentWorkHistory/phase6/2026-05-12-phase-6-closure-landscape-diagnostic.md
```

For the implementation surface, inspect:

```text
src/model.py
src/data.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase6_sum_only_semantic_decoder_gate.py
scripts/run_phase6_closure_landscape_diagnostic.py
```

## Critical Guardrails

Do not rediscover oracle success.

Oracle operands, oracle-at-eval recovery, forced true result classes,
injection-zero controls, and forced-random controls are wiring checks only.
They must pass before learned failures are interpretable, but they are not
progress on the central question.

For Phase 7 natural `0..19`, the product answer decoder/readout usability is
already established. Treat this as settled infrastructure, not a discovery
target. Future tasks should not spend experiment budget re-proving that oracle
or forced-true calculator results can answer natural addition unless the
semantic decoder checkpoint, answer-decoder implementation, calculator output
format, or readout path has changed. If such a regression check is strictly
needed after code changes, label it as a wiring regression only and immediately
return to learned-interface metrics.

Do not require exact true operand pairs as the main success metric for natural
sum-only addition.

For natural addition, `(3, 7)`, `(4, 6)`, `(5, 5)`, and other same-sum pairs
are answer-equivalent. Pair exact is still useful for diagnosis, but the Phase
7 learned-interface target is result-level calculator use:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- private all-pair result accuracy;
- full-enum learned-result best fraction;
- learned-result minus best-result answer-NLL gap;
- answer dependence on the learned calculator result under injection-zero and
  forced-random interventions;
- semantic decoder movement exactly `0.0`;
- auxiliary/direct operand supervision exactly `0.0`;
- relaxation/local/result-group objective exactly `0.0` for retention claims.

Do not start with `operand_max=99`, three-digit arithmetic, or natural language
tasks. Phase 7 should solve natural `0..19` result-level discovery first.

## Key Interpretation From Prior Phases

Phase 1 established that answer accuracy alone is unreliable: models can solve
around the calculator, use private codes, or benefit from oracle wiring without
learning the intended interface.

Phase 2 and Phase 3 showed that answer-NLL landscapes often contain real
calculator-action signal, but addition-only pair targets are broad and sampled
or independent-head objectives did not robustly discover hard protocols.

Phase 4 showed that when the answer target makes operand identity useful,
direct protocol teaching can be retained after direct supervision is removed.

Phase 5 showed that upstream movement can preserve or complete partially taught
protocols, but plain answer-only training does not discover the protocol from
scratch.

Phase 6 showed that answer-derived interface discovery is possible in the
identifiable setting, and that deterministic Concrete is the strongest current
estimator. It also showed that natural sum-only failure is best explained by
result-level underidentification plus independent-head action parameterization.

Phase 7 should treat that diagnosis as the starting point, not as a reason to
rerun Phase 6 schedules.

## Fixed Phase 7 Baseline Setup

Unless a task explicitly says otherwise, keep the natural `0..19` setting fixed:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
answer_format=sum
calculator_output_format=sum
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
answer_decoder_interaction=product
freeze_semantic_decoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
oracle_train=false
oracle_warmup_steps=0
```

Use the Phase 6 product-decoder checkpoint as the standard natural semantic
decoder gate:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

If that path is absent in a sandbox, use the exact path recorded in
`factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`.

For strict discovery branches, prefer:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
```

## Primary Research Tracks

## Current Training Approach Roadmap

The next work is not another oracle/readout check and not a small schedule
sweep of the failed joint-pair setup. Phase 7 should now use the following
decision tree:

1. **Result-space interface diagnostic.** Train a `0..38` result request head
   with answer loss, then map the predicted result to a deterministic valid
   calculator query. This asks whether the model can learn the result-level
   calculator request when the action space matches the variable natural
   answer loss actually identifies.
2. **Objective-off retention.** If result-space Stage 1 passes or strongly
   near-passes, continue with hard result requests and all relaxation/local/
   auxiliary/expected/anchor objectives exactly off. This tests whether the
   learned request survives after the training bridge is removed.
3. **Canonical query symmetry breaker.** If result-space learning works,
   convert result requests into stable calculator-query protocols by imposing
   one deterministic query convention per result. This is the bridge from
   "learn the result" toward "learn a calculator call."
4. **Replication.** Replicate only after the result-space or canonical-query
   branch passes both discovery and retention for seed `2`.
5. **New estimator families.** If result-space learning fails from strict
   initialization, stop treating action parameterization as the main blocker
   and move to qualitatively different training signals: policy-gradient /
   REINFORCE-style calculator actions, target-propagation or local boundary
   targets, differentiable surrogate/shadow-calculator gradients, synthetic
   gradients/direct feedback alignment, or explicit curricula with teacher
   removal.

Interpret result-space carefully: it is a diagnostic floor, not the final
calculator-query claim. A positive says result-level calculator requests are
learnable; a negative says the blocker is deeper than pair underidentification
or joint-pair optimization.

## Status Update: 2026-05-14

Phase 7 now has two natural `0..19` exact-grid results:

```text
exact_grid_seed_replication_negative
multisample_result_space_policy_gradient_stage0_alignment_negative
```

Exact full-grid upstream-open result-boundary teaching works: seed `2` learned
a hard result request to `0.9675`, and CLI seeds `4` and `5` relearned Stage 1
requests to `1.0000` and `0.9975`. However, strict target-off retention did
not robustly replicate. Seeds `4` and `5` retained only `87.0%` and `88.2%` at
their best post-start checkpoints, below the `90%` gate. Semantic decoder
movement stayed exactly `0.0`.

Useful prior negatives remain informative:

- joint-pair and direct result-space strict Concrete did not learn from
  initialization;
- frozen linear and frozen MLP result-boundary heads did not pass Stage 1;
- random-resampled upstream-open boundary teaching was only partial;
- exact-grid coverage stabilized that partial branch into a single-seed
  retained positive, but strict retention did not robustly replicate;
- vanilla `K=16` result-space REINFORCE produced nonzero result-proj/upstream
  gradients, but its Stage 0 sampled gradient was anti-aligned with the
  boundary-target ceiling (`result-proj cosine=-0.0945`, upstream
  `cosine=-0.1108`).
- exact result-marginal expected answer-loss over result classes also failed
  the Stage 0 alignment gate. Raw exact expected-cost gradients were nonzero
  but anti-aligned with the boundary-target ceiling (`result-proj
  cosine=-0.0978`, upstream `cosine=-0.1231`), and sampled PG was strongly
  aligned with that raw exact gradient. Detached z-score normalization weakly
  improved the result-head cosine but still failed upstream-open alignment.

Current next direction:

```text
Stop treating boundary-target teaching/retention, vanilla sampled
policy-gradient, and raw exact expected-cost training as the mainline. Use the
boundary-target branch as a supervised ceiling/control for new mechanisms, and
pivot to a qualitatively different learning signal. The selected next task is
the gradient-friendly result decoder alignment gate: test whether downstream
decoder/loss geometry can make the exact answer-loss gradient over result
actions align with the boundary-target ceiling before spending long-run
training budget. If that fails, move to explicitly biased backward channels
such as synthetic gradients/direct feedback alignment or learned
shadow-gradient modules.
```

Selected next task:

```text
aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
```

### Track A: Structured Joint-Pair Result-Group Bridge

This was the original Phase 7 mainline. After the exact-grid seed-replication
negative, do not treat it as the immediate next task unless the selected
policy-gradient gate creates a reason to return to joint-pair optimization.

Replace independent A/B operand-head pressure with a structured pair policy
that can put probability mass on a same-result group.

The preferred implementation is:

- a joint `20 x 20` pair policy over calculator calls;
- hard forward through the real calculator using the selected pair;
- soft backward through a result distribution formed by summing joint-pair mass
  over all pairs with the same calculator result;
- answer loss through the frozen product decoder;
- dense checkpoints and relaxation-off retention.

The expected successful behavior is not a unique true pair. It is a hard pair
whose calculator result is correct, with causal answer dependence on that
calculator result.

Implementation note: the repo already has `calculator_action_head=joint_pair`,
but current CLI validation restricts it to
`action_loss_full_enum_joint_interface`. Phase 7 may need to extend joint-pair
support to deterministic Concrete / result-group relaxed training and to the
diagnostics that read learned result groups.

### Track B: Result-Space Interface Diagnostic

This is the fastest diagnostic fallback, not the strongest final claim.

Train a `0..38` natural result-space interface and map each predicted result to
a valid calculator query, for example a deterministic canonical pair within
`0..19`.

This asks:

```text
Can answer loss learn a calculator-result request at all when the interface
action matches the identified variable?
```

A result-space positive is useful, but weaker than Track A because it risks
becoming answer-class prediction wrapped around a calculator call. If used,
require strict causal controls:

- hard calculator result accuracy;
- injection-zero and forced-random near chance;
- oracle-at-eval as wiring only;
- no semantic decoder movement;
- retention after any result-space relaxation or local objective is off.

### Track C: Canonical Query Symmetry Breaker

Use this only if Track A result-group training is unstable despite a positive
gradient gate.

Construct an answer-derived canonical query target from the best result group
without using true operands. Examples:

- fixed canonical representative for each result, such as the smallest valid
  first operand;
- entropy-regularized group target that later anneals to a stable
  representative;
- teacher selected from the answer-NLL best-result group plus a deterministic
  tie-break rule.

This branch can produce hard calculator calls, but claims must be labeled
carefully: it solves natural result use with an imposed query convention, not
true operand recovery.

### Track D: Upstream-Open Natural Result Discovery

Do not start here.

Only open upstream after a frozen-upstream or input-proj-only result-level
interface passes Stage 1 and Stage 2. When upstream opens, require dense
checkpointing and report parameter deltas for input projection, upstream, and
semantic decoder.

## Phase 7 Standard Stages

### Stage 0: Natural Decoder And Landscape Gate

Before training a new natural interface, verify:

- product decoder oracle-at-eval is `1.000` or near exact;
- injection-zero and forced-random are near chance;
- full-enum best-result group matches the true sum;
- answer-derived pair target remains broad, confirming that pair exact should
  not be the primary natural metric;
- result-group target is sharp enough to train against.

Useful gates:

```text
oracle-at-eval exact >= 0.99
best-result group matches true sum >= 0.99
mean soft target true-result group probability high
mean soft target true-pair probability low or broad is acceptable
semantic decoder delta exactly 0.0
```

### Stage 1: Result-Level Discovery Training

Train the selected interface without true operand labels:

```text
aux_operand_loss_weight=0.0
oracle_train=false
oracle_warmup_steps=0
freeze_semantic_decoder=true
```

Primary success metrics:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- normal answer exact;
- private all-pair result accuracy;
- full-enum learned-result best fraction;
- learned-result minus best-result gap.

Save dense checkpoints every `25` or `50` steps. Select by result-level
protocol metrics, not final eval exact alone.

### Stage 2: Relaxation-Off Or Objective-Off Retention

From selected Stage 1 checkpoints, continue with the real hard calculator path
and all discovery-specific objectives off:

```text
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
```

This is the central retention claim. Stage 1 alone is not enough.

### Stage 3: Replication

After one seed passes Stage 2, replicate across effective seeds `2`, `4`, and
`5`, matching Phase 6's replication discipline.

### Stage 4: Optional Upstream-Open Stress

Only after replication, test whether modest upstream movement helps or harms
natural result discovery. Report dense snapshots and parameter deltas.

## Required Diagnostics

For every selected checkpoint, run or extend the canonical diagnostics to
report:

- built-in eval exact;
- normal exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- oracle-at-eval exact;
- learned operand exact, for diagnosis only;
- learned pair exact, for diagnosis only;
- result-equivalent pair accuracy;
- learned calculator-result accuracy;
- private all-pair answer exact;
- private all-pair result accuracy;
- full-enum learned result NLL, true result NLL, and best result NLL;
- learned-result minus best-result gap;
- learned-result best fraction;
- best-result group true-sum fraction;
- pair entropy and result entropy when a pair policy is used;
- trainable parameter groups;
- semantic decoder parameter delta;
- upstream and input-proj parameter deltas;
- exact final objective weights.

## Success Definitions

### Useful Stage 0 Positive

The natural product decoder is healthy and the answer-derived result landscape
is sharp, while the pair landscape remains broad enough to justify a
result-group objective.

### Useful Stage 1 Positive

Without direct operand supervision or oracle operands, the learned hard
calculator result rises materially above chance and the full-enum learned-result
gap closes.

### Strong Phase 7 Positive

After relaxation or result-group teaching is off, answer-only continuation
retains a natural hard calculator-result protocol:

```text
normal answer exact near exact
learned calculator-result accuracy near exact
private all-pair result accuracy near exact
full-enum learned-result best fraction near 1.0
learned-result minus best-result gap near 0.0
injection-zero and forced-random near chance
semantic decoder delta 0.0
aux/direct operand supervision 0.0
all discovery-specific objective weights 0.0
```

### Very Strong Phase 7 Positive

The strong result replicates across effective seeds `2`, `4`, and `5`, and an
upstream-open branch preserves or improves result-level protocol metrics without
moving the semantic decoder.

## Failure Interpretation

If Stage 0 oracle/readout gates fail:

```text
Do not train the learned interface. Fix the natural product decoder or readout
first.
```

If Stage 0 result group is sharp but Stage 1 fails:

```text
The result-level objective is informative, but the action parameterization or
optimizer is still wrong. Prefer changing the interface parameterization before
rerunning small schedule sweeps.
```

If Stage 1 succeeds but Stage 2 retention fails:

```text
The training bridge can teach natural result use, but answer-only hard
calculator continuation cannot yet hold it. Investigate gate-triggered handoff,
slower relaxation removal, or stability regularization.
```

If result-space succeeds but joint-pair fails:

```text
The task is learnable at result level, but pair-action policy optimization is
the blocker. Use the result-space branch as a diagnostic floor, not the final
calculator-query claim.
```

## Reporting Contract

Every Phase 7 task completion should update or create:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-<short-description>.md
```

Every completed task should include:

- claim tested;
- code changes;
- exact commands;
- run paths;
- selected checkpoints;
- target/result landscape summary if relevant;
- fast-gate table;
- full diagnostics table;
- exact final objective weights;
- parameter movement summary;
- comparison to Phase 6 natural and identifiable baselines;
- go/no-go recommendation.

When a task is complete, move the task file to:

```text
aiAgentProjectTasks/completed/phase7/
```

then commit and push, following `CLAUDE.md`.

## First Task Recommendation

The first Phase 7 task should be:

```text
Phase 7 First Task: Natural Joint-Pair Result-Group Bridge Gate
```

It should:

1. Verify `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md` exists and add
   `aiAgentWorkHistory/phase7/` when writing the first work history.
2. Extend joint-pair diagnostics to summarize result-group mass and learned
   result accuracy under natural sum-only addition.
3. Add the smallest implementation needed for a joint-pair result-group
   deterministic Concrete bridge, or prove with a gradient gate that the
   existing implementation cannot support it cleanly.
4. Run Stage 0 product-decoder and full-enum result landscape gates.
5. Run one strict `semantic_decoder_only`, frozen-upstream Stage 1 seed only if
   the gates pass.
6. Proceed to retention and replication only after result-level hard protocol
   metrics pass, not merely after answer exact improves.

## Status Update: 2026-05-28

The first biased backward-channel branch has been tested:

```text
boundary_feedback_stage0_output_projection_alignment_pass_stage1_discovery_negative
fixed_random_direct_feedback_stage0_result_head_alignment_negative
```

Output-projection boundary feedback passed the Stage 0 exact-grid alignment
gate against the boundary-target ceiling (`result-proj cosine=0.2772`,
upstream `0.4382`, semantic decoder gradient `0.0`). However, Stage 1 training
with semantic decoder frozen and all oracle/boundary/aux/expected objectives
off reached only `0.155` best snapshot calculator-result accuracy and `0.160`
final exact match. A fixed-random direct-feedback seed had positive upstream
cosine but failed the result-head gate (`-0.0036`), so no long random-DFA run
was launched.

Current next direction:

```text
learned shadow-gradient / synthetic-gradient module with Stage 0 gate and early
Stage 1 lift requirement
```

Do not rerun plain output-projection feedback or fixed-random DFA long runs
without a genuinely stronger feedback mechanism and a positive result-head
Stage 0 gate.

## Status Update: 2026-05-28, Linear Shadow Feedback

A fit-once linear shadow-feedback branch was tested after the boundary-feedback
negative:

```text
linear_shadow_feedback_stage0_alignment_pass_stage1_early_lift_negative
```

The Stage 0 diagnostic fit a linear map from answer-loss injection gradients to
boundary result-logit gradients. It passed at the model-update level
(`result-proj cosine=0.9983`, upstream `0.9854`, semantic decoder gradient
`0.0`), but the frozen-map 200-step Stage 1 smoke failed to show early lift
(`0.070` best snapshot calculator-result accuracy, `0.040` final exact match).

Do not run an 800-step continuation of this exact setup. The next shadow
branch must improve on this by using heldout validation and/or online training
of the shadow module, with an early Stage 1 lift gate before long-run budget.

## Status Update: 2026-05-28, Heldout Linear Shadow Feedback

The same-batch linear shadow-feedback Stage 0 gate was tested under a
deterministic `320/80` exact-grid fit/heldout split:

```text
heldout_linear_shadow_feedback_stage0_generalization_negative
```

Train result-proj cosine remained near perfect (`0.9981`), but heldout
result-proj cosine fell to `0.2622`, with a `0.7359` train-heldout gap.
Heldout upstream cosine was `0.5101`, relative norms were close to `1.0`, and
semantic decoder gradient remained `0.0`.

Do not use same-batch linear shadow alignment as a go gate. The next task
should be online MLP shadow feedback with result-policy state, heldout warmup
validation, and a 200-step early-lift gate before any long run.

## Status Update: 2026-05-28, Online MLP Shadow Feedback

The first online MLP shadow-feedback warmup gate was implemented and tested:

```text
online_mlp_shadow_feedback_stage0b_partial_alignment_no_clean_gate
```

The online MLP uses per-example-scaled answer injection gradients plus current
result logits as input state, trains only the shadow module during warmup, and
then compares the induced model gradients with the boundary-target ceiling on
train and heldout splits.

Hidden size `64` produced promising heldout alignment (`0.7167` result-proj
cosine, `0.7601` upstream cosine), but train-heldout gaps were too large
(`0.2683` result, `0.2202` upstream). Hidden size `16` reduced the result gap
but missed the result-proj threshold (`0.6255` heldout cosine).

No Stage 1 early-lift run was launched. The next shadow-gradient task should
add a real generalization improvement such as validation early stopping,
regularization, target normalization, richer policy state, or a different
synthetic-gradient objective before spending Stage 1 budget.

## Status Update: 2026-05-28, Online MLP Validation Selection

Validation-selected checkpointing was added to the online MLP shadow-feedback
diagnostic:

```text
online_mlp_shadow_feedback_validation_selection_negative
```

The diagnostic now keeps a separate validation split for shadow checkpoint
selection and an untouched heldout-test split for the final gate. With `h64`,
`lr=1e-3`, `100` warmup steps, `0.1` validation, and `0.2` heldout test, the
best validation checkpoint was step `60`. It reached heldout-test
result/upstream cosines `0.6449/0.7266`, with train-test gaps
`0.3201/0.2414`. The final unselected checkpoint was closer on result cosine
(`0.6955`) but still below the `0.70` result threshold.

No Stage 1 early-lift run was launched. Next work should change the shadow
target/state or add stronger regularization rather than relying on validation
selection alone.

## Status Update: 2026-05-28, Online MLP Target Normalization

Fit-split per-result z-score target normalization was added to the online MLP
shadow-feedback diagnostic:

```text
online_mlp_shadow_feedback_target_normalization_partial_no_go
```

The diagnostic now fits target mean/std on the fit split only, trains the MLP
on normalized targets, and unnormalizes predictions before raw model-gradient
agreement checks. This improved heldout-test alignment but still did not clear
the complete gate. Best near miss was hidden size `16`: heldout-test
result/upstream cosines `0.7259/0.7549`, relative norms `1.4146/1.1848`, and
train-heldout gaps `0.1723/0.1458`.

No Stage 1 early-lift run was launched. Next work should change shadow
input/state or objective more substantially, not rerun this exact
target-normalized sweep.

## Status Update: 2026-05-28, Online MLP Policy-State Features

Richer raw policy-state features were added to the target-normalized online
MLP shadow-feedback diagnostic:

```text
online_mlp_shadow_feedback_policy_state_raw_features_negative
```

The policy-state feature mode appends result probabilities,
log-probabilities, and entropy to the existing answer-gradient plus result
logit state. It did not clear the heldout gate. Hidden size `32` reached
heldout-test result/upstream cosines `0.7037/0.7611`, but train-heldout gaps
widened to `0.2853/0.2131`. Hidden size `16` missed the result threshold
(`0.6862`).

No Stage 1 early-lift run was launched. Next work should address feature
scaling, regularization, loss shape, or target construction rather than simply
appending raw policy features.
