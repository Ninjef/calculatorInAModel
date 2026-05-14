# Phase 7 Tenth Task: Gradient-Friendly Result Decoder Alignment Gate

## Mission

Phase 7 has now learned a sharper lesson than "the estimator is noisy":

```text
The exact expected answer-loss gradient over natural result actions is locally
anti-aligned with the known good boundary-target direction for the current
frozen product decoder.
```

That means vanilla score-function methods, learned baselines that preserve the
same expectation, and raw exact expected-cost training are not the fastest next
path. The current decoder can answer correctly when forced the true result, but
its answer-loss geometry does not currently provide a useful model-side result
request gradient at initialization.

The next task should ask:

```text
Can we make the downstream result decoder not merely accurate under forced
true results, but gradient-friendly enough that the exact answer-loss objective
over result actions aligns with the boundary-target ceiling?
```

If yes, then run result-space discovery with the aligned decoder frozen. If no,
stop and pivot to explicitly biased backward signals such as synthetic
gradients/direct feedback alignment or learned shadow-gradient modules.

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
aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md
aiAgentProjectTasks/completed/phase7/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
aiAgentProjectTasks/completed/phase7/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md
```

For implementation, inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_sum_only_semantic_decoder_gate.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_calculator_protocol.py
tests/test_model.py
```

## Why This Is The Next Best Task

Helpful knowledge from Phase 7:

- Exact full-grid coverage is now the right default for natural `0..19`
  result-interface gates.
- The boundary-target branch proves the natural hard result request is
  representable and teachable with upstream movement and semantic decoder
  movement exactly `0.0`.
- The boundary-target gradient is a useful supervised ceiling/control for
  future mechanisms, even though boundary-target retention itself was
  seed-fragile under the strict gate.
- Result-space REINFORCE plumbing and exact result-marginal expected-loss
  plumbing are valuable diagnostics, even though their raw objective failed.
- The exact result-marginal gate removed the variance ambiguity: sampled PG was
  strongly aligned with the raw exact expected-cost gradient, and both were
  anti-aligned with the boundary-target ceiling.

Less helpful as next work:

- Oracle/readout reruns for the existing Phase 6 product decoder.
- More random-resampled or frozen-head boundary-target variants.
- More target-off retention reruns that do not introduce a new mechanism or
  directly diagnose the observed fragility.
- Canonical-query stabilization before a robust natural result request exists.
- Longer vanilla result-space PG schedules, actor-critic baselines, or
  RELAX/NVIL variants that only estimate the same raw expected-cost gradient.
- Rerunning raw exact result-marginal expected-cost training with the same
  frozen decoder.

The fastest honest fork is now decoder geometry:

```text
If a gradient-friendly decoder makes exact answer-loss gradients align, then
the project can keep an answer-derived discovery story and move immediately to
Stage 1 result-space learning.

If no decoder variant can pass the fixed-grid alignment gate, then Phase 7
should stop trying to make ordinary answer loss carry the boundary signal and
move to explicitly learned/biased backward channels.
```

## Required Interpretation

Do not count oracle decoder success as research progress.

Decoder pretraining is infrastructure. It is allowed to use oracle or forced
true result inputs because Phase 7 already depends on a frozen downstream
decoder. The substantive claim begins only when the upstream/model-side result
request learns with the decoder frozen and without true-result, true-operand,
boundary-target, or oracle calculator-action supervision.

The new decoder must be judged by gradient alignment before any long training:

```text
forced true result works is necessary but not sufficient.
exact expected answer-loss vs boundary-target gradient alignment is the gate.
```

## Claim Tested

Primary diagnostic claim:

```text
A result-calibrated frozen answer decoder can make the exact result-marginal
answer-loss gradient over result actions positively aligned with the
answer-derived boundary-target ceiling on the exact natural `20 x 20` grid.
```

Primary training claim, only if the diagnostic passes:

```text
With such a decoder frozen, exact result-marginal answer-loss training can
discover a hard model-side calculator-result request without oracle operands,
true result labels, boundary-target CE/KL, sampled policy-gradient updates,
direct operand labels, or semantic decoder movement.
```

## Stage 0: Decoder Candidate And Gradient Alignment Gate

Implement the smallest runner or extension needed to train and evaluate one or
two result-calibrated decoder candidates.

Use the fixed Phase 7 natural setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
calculator_result_vocab_size=39
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
answer_format=sum
calculator_output_format=sum
calculator_bottleneck_mode=answer_decoder
answer_decoder_interaction=product
calculator_action_head=result_space
calculator_read_position=operand_spans
calculator_read_span_width=2
oracle_train=true only during decoder pretraining
semantic_decoder_checkpoint_load_scope=semantic_decoder_only for downstream tests
freeze_semantic_decoder=true for downstream tests
```

Candidate decoder training should improve loss geometry, not just forced-result
accuracy. Start with at most two branches:

1. **Soft-result calibration branch.**
   Train the answer decoder on oracle/forced true results plus noisy soft
   result distributions around the true result. The goal is for the decoder to
   learn a smooth local answer-loss slope from broad result policies toward the
   correct result.
2. **Contrastive result-margin branch.**
   Train the answer decoder so the true result class has a strong answer-NLL
   margin against forced wrong result classes. Use forced wrong results for the
   decoder-only loss, not for model-side interface training.

Do not run a broad decoder architecture sweep. Keep the product decoder shape
unless the current code makes a small decoder-only variant cheaper than adding
the calibration losses.

For each candidate, run the same exact-grid diagnostic discipline used by the
ninth task:

- exhaustive `20 x 20` grid, `400` prompts;
- `calculator_action_head=result_space`;
- `calculator_estimator=full_enum_expected_answer_loss`;
- semantic decoder frozen;
- upstream open for the gradient measurement;
- no boundary-target update;
- no oracle operands during the downstream diagnostic;
- compute exact result-marginal expected-loss, sampled result-space PG, and
  boundary-target gradients on the same batch.

Report, for the existing baseline decoder and every new candidate:

```text
forced-true/oracle exact accuracy
hard-best result equals true sum
tie-aware true-result best fraction
raw expected NLL
best/true result NLL
learned result NLL
expected-minus-best gap
exact result-proj grad L2
exact upstream grad L2
semantic decoder grad L2
exact-vs-boundary result-proj cosine
exact-vs-boundary upstream cosine
sampled PG-vs-exact cosine
sampled PG-vs-boundary cosine
```

Stage 0 passes only if one candidate satisfies:

```text
forced-true/oracle exact accuracy remains high
hard-best result equals true sum >= 0.99 on the exact grid
exact result-proj grad L2 > 0
exact upstream grad L2 > 0
semantic decoder grad/delta L2 == 0.0 in downstream diagnostic
exact-vs-boundary result-proj cosine > 0.0
exact-vs-boundary upstream cosine > 0.0
```

Prefer a stronger threshold such as `> 0.10` cosine for both groups if more
than one candidate passes.

If no candidate passes Stage 0, stop. Do not run long discovery training.
Record the result as:

```text
gradient_friendly_decoder_alignment_negative
```

## Stage 1: Result-Marginal Discovery With The Aligned Decoder

Run only if Stage 0 passes.

Initialize from the selected decoder checkpoint and freeze the semantic decoder.
Use exact-grid result-marginal answer-loss training:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
calculator_result_vocab_size=39
calculator_action_head=result_space
calculator_estimator=full_enum_expected_answer_loss
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
answer_decoder_interaction=product
semantic_decoder_checkpoint=<selected gradient-friendly decoder checkpoint>
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
expected_answer_loss_policy_temperature=1.0
expected_answer_loss_cost_normalization=none
expected_answer_loss_entropy_weight=0.0
expected_answer_loss_chunk_size=64
freeze_upstream_encoder=false
input_proj_lr=0.01
upstream_lr=0.0003
steps=800
snapshot_every=25
checkpoint_every=25
```

Allowed rescues, only if Stage 0 alignment was clearly positive and the
primary branch moves meaningfully but misses the hard-result gate:

1. `expected_answer_loss_cost_normalization=zscore`
2. `expected_answer_loss_policy_temperature=0.5`

Run at most one rescue unless the first rescue is close to the pass threshold.

Stage 1 checkpoint selection must use learned-interface metrics:

- hard learned calculator-result accuracy;
- full-enum learned-result best fraction;
- mean learned-result minus best-result gap;
- expected-minus-best NLL gap;
- result entropy/effective results;
- canonical normal exact and calculator-result accuracy;
- injection-zero, forced-random, and oracle-at-eval controls;
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

If Stage 1 does not pass, stop and record:

```text
gradient_friendly_decoder_expected_loss_stage1_negative
```

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
SOLUTION_IDEAS.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
```

Record:

- code changes and validation commands;
- decoder candidate training setup;
- Stage 0 candidate table with forced-result, exact expected-loss, sampled PG,
  and boundary gradient norms/cosines;
- Stage 1 branch table if run;
- Stage 2 retention table if run;
- final decision label.

Use one of these final labels:

```text
gradient_friendly_decoder_alignment_negative
gradient_friendly_decoder_expected_loss_stage1_negative
gradient_friendly_decoder_expected_loss_retention_negative
gradient_friendly_decoder_expected_loss_retained_positive
```

## Stop Rules

Stop before long training if no decoder candidate passes the exact-grid
gradient-alignment gate.

Stop before Stage 2 if Stage 1 hard learned calculator-result accuracy is below
`0.70`.

Do not spend time on:

- oracle-only claims;
- independent-head expected-answer-loss sweeps;
- vanilla result-space REINFORCE long runs;
- raw exact expected-cost reruns with the old decoder;
- boundary-target retention reruns;
- canonical-query stabilization before this task clarifies whether decoder
  geometry can rescue answer-loss discovery.

## Commit

After completing the task:

```bash
git status --short
git add src/model.py scripts/overfit_one_batch.py tests/test_model.py factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md SOLUTION_IDEAS.md aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md aiAgentWorkHistory/phase7/<work-history-file>.md aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
git commit -m "Run gradient-friendly decoder alignment gate"
git push
```
