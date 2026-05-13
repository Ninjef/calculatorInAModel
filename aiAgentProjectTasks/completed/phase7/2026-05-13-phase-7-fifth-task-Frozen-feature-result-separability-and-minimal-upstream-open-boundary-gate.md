# Phase 7 Fifth Task: Frozen-Feature Result Separability And Minimal Upstream-Open Boundary Gate

## Claim

Phase 7 has now produced three strict natural `0..19` negatives:

```text
joint_pair_stage1_negative:
  hard learned calculator-result accuracy peaked at 0.1100
  soft true-result probability stayed near broad initial mass

result_space_stage1_negative:
  direct 0..38 result request accuracy peaked at 0.0925
  soft true-result probability moved only 0.02564 -> 0.02920

result_boundary_target_stage1_negative:
  answer-derived best-result targets were sharp and valid
  frozen result_proj teaching still peaked at only 0.1150
```

The next task should answer the question those negatives now put directly on
the table:

```text
Do the frozen operand-span representations contain enough information for a
linear or shallow result request head to recover the answer-derived calculator
result target, or must Phase 7 let the upstream representation move?
```

This is a feature/capacity gate plus the smallest useful upstream-open rescue.
It is not a new oracle/readout check and not another deterministic Concrete
schedule sweep.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 6 proved that answer-derived local targets can teach and then hand off
  a calculator protocol in an identifiable setting.
- Phase 6/7 natural gates proved that the product answer decoder and forced
  result landscape are healthy: the answer-derived best result class is the
  true sum essentially always.
- The Phase 7 result-space head removed pair underidentification, but answer
  loss still did not teach the request.
- The Phase 7 result-boundary target removed downstream credit ambiguity even
  further: it directly trained `calculator_hook.result_proj` toward the
  answer-derived best result. That still failed with frozen upstream features.
- Therefore the current highest-value uncertainty is no longer decoder health
  or target quality. It is frozen feature availability, head capacity, or the
  need for representation learning.

Less helpful to repeat now:

- Oracle operands, oracle-at-eval recovery, forced true result, forced random,
  or injection-zero checks unless a code change could have broken wiring. Those
  are already settled regression checks.
- Small LR, temperature, entropy, or step-count sweeps of the frozen linear
  `result_proj` branch. The direct boundary target already behaved like a
  supervised result classifier and did not overfit.
- Seed replication or retention from any Phase 7 negative checkpoint.
- Canonical-query symmetry breaking or `operand_max=99` scaling before the
  model can learn the natural `0..19` result request.
- Treating exact true operand-pair recovery as the success target for natural
  sum-only addition.

The fastest path to the end goal is now:

1. Test whether the exact frozen features consumed by `result_proj` can support
   the answer-derived result target under a controlled probe.
2. If a linear probe succeeds, find why the in-model boundary target did not.
3. If only a shallow probe succeeds, try the smallest MLP result head under the
   same boundary target.
4. If frozen probes fail, allow upstream movement under the boundary target and
   immediately test target-off retention if teaching works.

This keeps the project moving toward learned calculator use while avoiding
another expensive pass through already-failed frozen-head recipes.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-interface-diagnostic.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-boundary-target-learning-signal.md
```

Inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Fixed Setup

Use the Phase 7 natural `0..19` result-space setup unless a substage explicitly
says otherwise:

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
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
```

Use the Phase 6 product-decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Use this run root:

```text
runs/2026-05-13_phase7_result_feature_separability_and_upstream_open
```

## Part 1: Add A Result Separability Diagnostic

Add a narrow diagnostic script or extend `scripts/diagnose_calculator_protocol.py`
with a result probe mode. Prefer a new script if that keeps the probe output
clean:

```text
scripts/run_phase7_result_feature_separability.py
```

Required probe dataset:

- Build the same strict `semantic_decoder_only` model used by Phase 7.
- Generate the exhaustive natural `0..19` grid, `400` examples.
- For each example, collect the exact paired feature vector consumed by
  `calculator_hook.result_proj` under `calculator_read_position=operand_spans`.
- Also collect optional comparison features:
  - raw operand `a` span feature;
  - raw operand `b` span feature;
  - concatenated final-layer operand-span residuals;
  - calculator read residual already exposed in diagnostics, if useful.
- Construct the primary label from the answer-derived boundary target:
  enumerate forced result classes through the frozen product decoder and select
  the lowest answer NLL class.
- Compute the true sum only after target construction, for parity and
  interpretation.

Required probes:

- Linear probe over the exact `result_proj` input feature.
- Shallow MLP probe over the exact `result_proj` input feature:
  one hidden layer is enough for the primary gate, for example hidden sizes
  `64` and `128`.
- Optional operand probes for `a` and `b`, to separate "operand identity is
  present" from "sum/result is linearly decodable."

Probe hygiene:

- Normalize features using train-split statistics only for held-out folds.
- Report all-400 train accuracy because the task distribution is finite and
  the prior Stage 1 runs were one-batch overfit experiments.
- Also report 5-fold held-out accuracy to catch brittle memorization.
- Use multiple probe seeds, at least `2`, `4`, and `5`, for probe initialization
  only. Do not treat this as Phase 7 learned-interface replication.

Output:

```text
result_target_parity_with_true_sum
linear_all400_accuracy
linear_5fold_mean_accuracy
linear_5fold_min_accuracy
mlp64_all400_accuracy
mlp64_5fold_mean_accuracy
mlp128_all400_accuracy
mlp128_5fold_mean_accuracy
operand_a_linear_accuracy
operand_b_linear_accuracy
confusion_by_result_class
feature_norm_summary
```

Write JSON and CSV artifacts under the run root, and record the exact checkpoint
and config used.

Expected tests:

- Probe feature extraction shape equals `2 * calculator_read_span_width * n_embd`
  for `result_space + operand_spans`.
- The answer-derived target equals the true sum on a controlled natural batch
  only after target construction.
- Linear probe can overfit a synthetic linearly separable feature/label fixture.
- CLI validation rejects unsupported probe head kinds and non-positive hidden
  sizes, steps, or folds.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase7_result_feature_separability.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

## Part 2: Interpret The Separability Gate

Use these decision thresholds:

```text
linear_all400_accuracy >= 0.98:
  The exact frozen feature is linearly sufficient. The failed boundary-target
  run is likely an in-model optimization, feature extraction, or training-loop
  issue. Debug that mismatch before trying new estimators.

linear_all400_accuracy < 0.98 and mlp128_all400_accuracy >= 0.98:
  The frozen feature has useful information, but the current linear result head
  is too weak. Proceed to Part 3A.

mlp128_all400_accuracy < 0.98:
  The frozen strict operand-span representation is not enough for this target
  under a small head. Skip further frozen-head training and proceed to Part 3B.
```

If the answer-derived target does not match true sum at `>= 0.98`, stop and
debug target construction or checkpoint loading. Do not interpret probe results
until target parity passes.

## Part 3A: Conditional Small MLP Result Head

Run this part only if the shallow probe passes and the linear probe fails.

Implement the smallest production-path head needed to test the capacity
hypothesis:

```text
--calculator-result-head-hidden-size 0 | 64 | 128
```

Semantics:

- `0` preserves the current linear `result_proj`.
- `64` or `128` makes the result-space action head a two-layer MLP.
- Keep the hard forward calculator path unchanged.
- Keep `calculator_action_head=result_space`.
- Keep semantic decoder and upstream frozen.

Stage 1 teaching:

```text
answer_loss_weight=0.0
result_boundary_target_loss_weight=1.0
result_boundary_target_mode=hard_best_result
result_boundary_target_temperature=0.25
calculator_result_head_hidden_size=best passing probe size
input_proj_lr=0.01
steps=600
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
```

Pass gate:

```text
hard learned calculator-result accuracy >= 0.70
learned-result best fraction >= 0.70
mean learned-result minus best-result gap <= 1.0
semantic_decoder_delta = 0.0
upstream_delta = 0.0
```

If Stage 1 passes, run target-off retention immediately:

```text
answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
aux/adaptive/expected/entropy/anchor weights all 0.0
semantic_decoder frozen
upstream frozen
steps=300
```

Retention pass gate:

```text
canonical calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
injection-zero and forced-random remain low
semantic_decoder_delta = 0.0
result_boundary_target_loss_weight = 0.0
```

If the MLP head fails Stage 1, do not sweep hidden sizes beyond `64` and `128`
in this task. Proceed to Part 3B or write the next task for upstream movement.

## Part 3B: Conditional Minimal Upstream-Open Boundary Target

Run this part if frozen probes fail, or if the MLP head fails Stage 1.

Goal:

```text
Test whether answer-derived result targets can shape the upstream
representation into one that issues correct natural result requests.
```

Setup:

```text
calculator_action_head=result_space
calculator_result_head_hidden_size=0
freeze_semantic_decoder=true
freeze_upstream_encoder=false
answer_loss_weight=0.0
result_boundary_target_loss_weight=1.0
result_boundary_target_mode=hard_best_result
result_boundary_target_temperature=0.25
input_proj_lr=0.01
upstream_lr=0.0003
steps=600
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
```

The semantic decoder must remain frozen. The central metric is learned
calculator-result accuracy, not answer exact alone.

Pass gate:

```text
hard learned calculator-result accuracy >= 0.70
learned-result best fraction >= 0.70
semantic_decoder_delta = 0.0
result_proj_delta > 0.0
upstream_delta > 0.0
```

If this passes, run target-off retention from the best checkpoint:

```text
answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=false
aux/adaptive/expected/entropy/anchor weights all 0.0
steps=300
```

Also run a stricter retention variant with upstream frozen from the same
checkpoint if the upstream-open retention passes. This separates "protocol is
stable once learned" from "protocol needs continual representation drift."

Retention pass gate:

```text
canonical calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
mean learned-result minus best-result gap <= 1.0
semantic_decoder_delta = 0.0
result_boundary_target_loss_weight = 0.0
```

If upstream-open boundary teaching fails below `0.30` hard result accuracy, do
not run retention. The next task should move to a different signal family:
multi-sample policy gradient with per-prompt or leave-one-out baselines,
surrogate gradients, or direct feedback alignment.

## Required Diagnostics For Any Selected Checkpoint

For the best Stage 1 checkpoint in Part 3A or Part 3B, run:

```text
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

Report:

- canonical normal exact;
- canonical calculator-result accuracy;
- result-equivalent pair accuracy;
- pair exact as diagnostic only;
- injection-zero exact;
- forced-random exact;
- oracle-at-eval exact as wiring regression only;
- full-enum learned-result best fraction;
- learned-result minus best-result answer-NLL gap;
- true result best fraction and tie-aware true result best fraction;
- semantic decoder parameter delta;
- result head parameter delta;
- upstream parameter delta when upstream is open;
- all final objective weights.

## Stop Conditions

Stop the task early if:

- answer-derived target parity fails;
- probe extraction does not match the exact in-model `result_proj` input;
- semantic decoder parameters move in any strict frozen-decoder branch;
- the branch under test is below `0.30` hard learned calculator-result accuracy
  after its planned Stage 1 budget.

Do not spend this task budget on:

- oracle-only claims;
- more frozen linear-head LR/temperature sweeps;
- result-space seed replication before a seed-2 pass;
- natural `0..99` scaling;
- natural language formatting;
- exact operand-pair recovery as the primary success metric.

## Deliverables

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/YYYY-MM-DD-result-feature-separability-and-upstream-open-boundary-gate.md
```

If the task is completed, move this task file to:

```text
aiAgentProjectTasks/completed/phase7/
```

Commit and push the implementation, diagnostics, fact sheet, work history, and
task-file move.
