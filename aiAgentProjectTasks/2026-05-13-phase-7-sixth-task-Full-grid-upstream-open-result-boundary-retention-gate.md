# Phase 7 Sixth Task: Full-Grid Upstream-Open Result Boundary Retention Gate

## Claim

Phase 7 has now narrowed the natural `0..19` failure to the model-side result
request interface, not to decoder/readout health:

```text
joint_pair_stage1_negative:
  hard learned calculator-result accuracy peaked at 0.1100

result_space_stage1_negative:
  direct 0..38 result request accuracy peaked at 0.0925

result_boundary_target_stage1_negative:
  answer-derived result targets were sharp, but frozen linear result_proj
  teaching peaked at 0.1150

frozen_feature_gate:
  frozen exact result_proj features were not linearly sufficient at threshold
  linear all-grid probe = 0.9217
  shallow MLP all-grid probe = 1.0000
  operand A/B linear probes = 1.0000

minimal_upstream_open_boundary_target_partial:
  semantic decoder delta = 0.0
  upstream-open hard result accuracy peaked at 0.5975
  final drifted down to 0.4275
```

The next task should test the most direct explanation of the near-miss:

```text
Can the answer-derived result boundary target teach the upstream-open
result-space interface when training is an exact full-grid overfit rather than
random resampling, and if it can, does the hard result request survive after
the boundary target is removed?
```

This is a stabilization and retention gate for the only Phase 7 natural branch
that has moved substantially above chance. It is not an oracle/readout rerun
and not another frozen-head or deterministic Concrete schedule sweep.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- The frozen product answer decoder is healthy: forced-result enumeration
  identifies the true sum essentially exactly, and oracle-at-eval recovery is a
  wiring check only.
- Direct result-space action parameterization removed same-sum pair ambiguity,
  but answer loss through deterministic Concrete did not teach the request.
- The answer-derived result boundary target is sharp and valid; true sums are
  not needed to construct it.
- Frozen features contain operand identity and nonlinear finite-grid result
  information, but the production frozen linear and frozen MLP heads did not
  pass Stage 1.
- Allowing upstream movement changed the result: hard result accuracy rose to
  `0.5975` with semantic decoder movement exactly `0.0`.
- The partial run drifted by final, which makes exact-grid coverage and dense
  checkpoint selection more valuable than a broad new family of hyperparameter
  sweeps.

Less helpful to do now:

- Oracle operands, oracle-at-eval, forced-true result, injection-zero, or
  forced-random checks except as post-change regression controls.
- More frozen linear/MLP result-head training under the same randomly resampled
  boundary-target setup.
- Seed replication or target-off retention from the existing `0.5975` partial
  checkpoint; it did not pass the Stage 1 gate.
- Canonical-query symmetry breaking or `operand_max=99` scaling before the
  natural `0..19` result request is learned reliably.
- Treating exact ordered operand-pair recovery as the primary success target
  for natural sum-only addition.

Strategic choice:

```text
Run one exact full-grid upstream-open stabilization gate before moving to
multi-sample policy gradient.
```

Reason: the upstream-open boundary target is currently the closest natural
result-level calculator-use path. If exact full-grid training makes it pass,
the project gets a short route to a target-off retention test. If it still
fails, the conclusion is cleaner: the problem is not random pair coverage or
stochastic minibatch noise, and Phase 7 should move to a genuinely different
signal family such as multi-sample policy gradient with per-prompt baselines.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-interface-diagnostic.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-boundary-target-learning-signal.md
aiAgentWorkHistory/phase7/2026-05-13-result-feature-separability-and-upstream-open-boundary-gate.md
```

Inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase7_result_feature_separability.py
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
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention
```

## Part 1: Add Exact Full-Grid Training Support

Add the smallest training-loop support needed to reuse the exact ordered
natural grid as the training batch.

Preferred CLI:

```text
--exhaustive-grid-batch
```

Behavior:

- Requires `--operand-max` to be set.
- Requires fixed-width prompts.
- Builds one deterministic batch containing every ordered pair
  `(a, b) in 0..operand_max x 0..operand_max` exactly once.
- For `operand_max=19`, the batch has exactly `400` examples.
- Reuses that same batch at every training step.
- Does not use true sums or operands for result-boundary target construction;
  the grid only controls prompt coverage.
- Evaluation and diagnostics may still compute true sums afterward for metrics.

Recommended implementation:

- Add a helper near `make_range_batch`, for example
  `make_exhaustive_range_batch(...)`.
- In `run_variant`, if `--exhaustive-grid-batch` is enabled, construct this
  batch once before the training loop and reuse it.
- Keep existing random `make_range_batch` behavior unchanged when the flag is
  absent.
- Record `exhaustive_grid_batch=true` and the grid size in `config.json` and
  `metrics.json`.

Expected tests:

- Exact-grid helper returns `(operand_max + 1) ** 2` examples.
- Every ordered pair appears exactly once.
- The helper pads and masks targets identically to `make_range_batch`.
- CLI validation rejects `--exhaustive-grid-batch` without `--operand-max`.
- A one-step smoke verifies the result-boundary target still has nonzero
  gradient into the trainable result/upstream groups and zero semantic decoder
  movement.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

## Part 2: Stage 0 Full-Grid Target Parity Gate

Before training, run a fixed full-grid parity check with the Phase 6 product
decoder checkpoint.

Report:

- hard-best result equals true sum;
- tie-aware true-result best fraction;
- soft target true-result probability;
- target entropy and effective result count;
- initial hard learned result accuracy;
- result-proj gradient L2;
- upstream gradient L2 when upstream is open;
- semantic decoder gradient/delta exactly `0.0`;
- exact grid coverage count and duplicate count.

Pass gate:

```text
grid_examples = 400
grid_duplicate_pairs = 0
hard_best_result_equals_true_sum >= 0.98
tie_aware_true_result_best_fraction >= 0.98
semantic_decoder_delta = 0.0
result_proj_gradient_l2 > 0.0
```

If this fails, stop and debug target construction, grid construction, or
checkpoint loading.

## Part 3: Stage 1 Exact-Grid Upstream-Open Teaching

Run seed `2` first.

Primary branch:

```text
exhaustive_grid_batch=true
answer_loss_weight=0.0
result_boundary_target_loss_weight=1.0
result_boundary_target_mode=hard_best_result
result_boundary_target_temperature=0.25
result_boundary_target_min_probability_floor=0.0
result_boundary_target_chunk_size=64
calculator_result_head_hidden_size=0
freeze_upstream_encoder=false
input_proj_lr=0.01
upstream_lr=0.0003
steps=800
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
batch_size=400
```

Use dense checkpoint selection. Select the best checkpoint by hard learned
calculator-result accuracy on the exact full grid, not by final-step accuracy.

Stage 1 pass gate:

```text
hard learned calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
semantic decoder delta = 0.0
injection-zero exact remains near prior control, roughly <= 0.10
forced-random exact remains near prior control, roughly <= 0.10
oracle-at-eval exact remains >= 0.98 as a regression check only
```

If the primary branch reaches `>=0.70`, skip rescue branches and proceed to
Part 5.

## Part 4: Single Capacity Rescue, Only If Needed

Run this only if the primary exact-grid branch peaks between `0.50` and `0.70`.
Do not run it if the primary branch is below `0.50`; that would mean exact-grid
coverage did not preserve the previous partial result.

Rescue branch:

```text
same as Part 3
calculator_result_head_hidden_size=64
steps=800
```

Rationale: the frozen MLP head did not pass by itself, but the separability
diagnostic showed shallow nonlinear all-grid capacity exists. This rescue asks
only whether upstream movement plus a minimal nonlinear result head crosses the
teaching gate under exact-grid coverage.

If this branch also fails the `0.70` Stage 1 gate, stop. Do not run retention
or seed replication. The next task should pivot to multi-sample result-space
policy gradient with per-prompt or leave-one-out baselines.

## Part 5: Stage 2 Target-Off Retention

Run only from the first Stage 1 checkpoint that passes the gate.

Continue from the selected checkpoint with every teaching objective off:

```text
answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=false
exhaustive_grid_batch=true
steps=400
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
```

This is a target-off retention check, not a new discovery claim by itself. The
hard calculator path must remain useful after the result-boundary target is
gone.

Retention pass gate:

```text
final hard learned calculator-result accuracy >= 0.70
final full-enum learned-result best fraction >= 0.70
best target-off checkpoint retains at least 90% of the selected Stage 1 hard
  result accuracy
semantic decoder delta = 0.0
result_boundary_target_loss_weight = 0.0
aux/adaptive/expected/anchor objectives = 0.0
injection-zero and forced-random controls stay low
```

If target-off retention collapses, label the result:

```text
full_grid_boundary_target_teaches_but_does_not_retain
```

If it passes, label it:

```text
full_grid_upstream_open_result_boundary_retained_positive
```

## Required Diagnostics

For each selected Stage 1 and Stage 2 checkpoint, run:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint <checkpoint> --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --output-dir <checkpoint_parent>/canonical_diagnostic
```

and:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint <checkpoint> --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root <checkpoint_parent>/full_enum_diagnostic
```

Also compute parameter deltas from Stage 1 step `0` to the selected checkpoint
and from the Stage 2 start checkpoint to the final/selected target-off
checkpoint:

```text
semantic decoder L2/max/changed tensors
calculator_hook.result_proj L2/max/changed tensors
upstream encoder L2/max/changed tensors
other interface groups L2/max/changed tensors
```

## Decision Rules

Use these labels:

```text
full_grid_upstream_open_boundary_target_negative:
  exact-grid primary and allowed rescue both fail Stage 1

full_grid_boundary_target_teaches_but_does_not_retain:
  Stage 1 passes, but target-off continuation fails retention

full_grid_upstream_open_result_boundary_retained_positive:
  Stage 1 passes and target-off continuation retains the hard result request
```

If the negative label is reached, the next task should not be another
boundary-target schedule or capacity variant. Move to:

```text
multi-sample result-space policy gradient with per-prompt or leave-one-out
baselines, using the frozen product decoder and hard calculator forward path.
```

## Reporting Requirements

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-full-grid-upstream-open-result-boundary-retention-gate.md
```

If the task is completed, move this file to:

```text
aiAgentProjectTasks/completed/phase7/
```

Commit and push.

Do not present oracle-at-eval, forced-true result, or decoder-readout checks as
new progress. They are regression checks only. The result that matters is
learned hard calculator-result behavior from the model-side request.
