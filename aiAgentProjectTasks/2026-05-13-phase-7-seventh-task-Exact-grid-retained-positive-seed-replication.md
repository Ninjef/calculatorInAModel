# Phase 7 Seventh Task: Exact-Grid Retained-Positive Seed Replication

## Claim

Phase 7 now has one natural `0..19` retained positive:

```text
full_grid_upstream_open_result_boundary_retained_positive
```

Seed `2` exact full-grid upstream-open result-boundary teaching learned a hard
result request to `0.9675`, and target-off continuation retained it at
`0.8325` final / `0.8800` best post-start with semantic decoder movement
exactly `0.0`.

The next task is to test whether this is robust across effective seeds, not to
try a new learning-signal family yet.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Exact ordered-grid coverage stabilized the only Phase 7 branch that had moved
  substantially above chance.
- The answer-derived result-boundary target is sharp and does not use true sums
  or true operands for target construction.
- Upstream movement appears necessary for the current natural positive, while
  semantic decoder movement stayed exactly `0.0`.
- The linear result head was sufficient under exact-grid upstream-open training;
  the planned MLP rescue was not needed.
- Injection-zero and forced-random controls stayed low, so the retained result
  request is causally useful rather than a decoder/readout artifact.

Less helpful to do now:

- oracle/readout checks for natural `0..19` except as regression controls;
- random-resampled upstream-open boundary-target repeats;
- frozen linear or frozen MLP result-head boundary teaching;
- the MLP rescue from the full-grid task;
- immediate policy-gradient or surrogate-gradient pivots before replication;
- canonical-query/protocol stabilization before confirming robustness;
- treating exact true operand-pair recovery as the primary natural metric.

Strategic choice:

```text
Replicate the exact-grid retained-positive recipe across seeds 4 and 5.
```

If seeds `2`, `4`, and `5` all pass discovery and target-off retention, Phase 7
has a robust natural result-level calculator-use positive and should move next
to canonical-query/protocol stabilization. If replication fails, do not spend
the next task on more small boundary-target schedule variants; compare against
multi-sample result-space policy gradient or another genuinely different
learning signal.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-13-full-grid-upstream-open-result-boundary-retention-gate.md
```

Inspect only as needed:

```text
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Fixed Setup

Use the exact seed-2 positive recipe unless a preflight check proves the code
changed:

```text
digits=2
operand_max=19
exhaustive_grid_batch=true
exhaustive_grid_size=400
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
calculator_result_head_hidden_size=0
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=1.0
relaxed_calculator_final_temperature=1.0
freeze_semantic_decoder=true
freeze_upstream_encoder=false
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.01
upstream_lr=0.0003
batch_size=400
eval_samples=400
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
```

Use the standard Phase 6 product-decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Use this run root:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_seed_replication
```

## Part 1: Preflight

Do not rerun the full Stage 0 oracle/readout gate unless code changed in the
calculator path, answer decoder, or full-grid batch construction.

Minimum preflight:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Confirm from the seed-2 artifact that the baseline remains:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage0_full_grid_parity_gate/stage0_full_grid_parity_summary.json
```

Expected baseline:

```text
grid_examples = 400
grid_duplicate_pairs = 0
hard_best_result_equals_true_sum = 1.0
tie_aware_true_result_best_fraction = 1.0
semantic_decoder_delta = 0.0
```

If those artifacts are absent, run only the parity gate needed to recreate
them; label it as a regression check, not progress.

## Part 2: Stage 1 Teaching Replication

Run seeds `4` and `5`. Do not rerun seed `2` unless the prior artifact is
missing or code changed.

Command template:

```bash
SEED=4
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 800 \
  --batch-size 400 \
  --eval-samples 400 \
  --answer-format sum \
  --answer-loss-weight 0.0 \
  --operand-max 19 \
  --exhaustive-grid-batch \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator gumbel_concrete_interface \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-output-format sum \
  --answer-decoder-interaction product \
  --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --freeze-semantic-decoder \
  --result-boundary-target-loss-weight 1.0 \
  --result-boundary-target-mode hard_best_result \
  --result-boundary-target-temperature 0.25 \
  --result-boundary-target-min-probability-floor 0.0 \
  --result-boundary-target-chunk-size 64 \
  --calculator-result-head-hidden-size 0 \
  --input-proj-lr 0.01 \
  --upstream-lr 0.0003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --snapshot-every 25 \
  --checkpoint-every 25 \
  --snapshot-samples 400 \
  --seed "$SEED" \
  --run-root runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_seed_replication/stage1_seed_${SEED}
```

Repeat with `SEED=5`.

Select the best Stage 1 checkpoint by hard learned calculator-result accuracy
on the exact full grid, not by final answer exact alone.

Stage 1 pass gate for each seed:

```text
hard learned calculator-result accuracy >= 0.70
full-enum learned-result best fraction >= 0.70
semantic decoder delta = 0.0
injection-zero exact <= 0.10
forced-random exact <= 0.10
oracle-at-eval exact >= 0.98 as a regression check only
```

If either seed fails Stage 1 below `0.70`, stop after diagnostics. Do not run
target-off continuation from a failed Stage 1 checkpoint.

## Part 3: Stage 2 Target-Off Retention Replication

For each seed that passes Stage 1, continue from its selected Stage 1
checkpoint with the result-boundary target exactly off.

Command template:

```bash
SEED=4
STAGE1_CKPT=<selected_stage1_checkpoint_for_seed_${SEED}>
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 400 \
  --batch-size 400 \
  --eval-samples 400 \
  --answer-format sum \
  --answer-loss-weight 1.0 \
  --operand-max 19 \
  --exhaustive-grid-batch \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator gumbel_concrete_interface \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-output-format sum \
  --answer-decoder-interaction product \
  --semantic-decoder-checkpoint "$STAGE1_CKPT" \
  --semantic-decoder-checkpoint-load-scope full_model \
  --freeze-semantic-decoder \
  --result-boundary-target-loss-weight 0.0 \
  --result-boundary-target-mode hard_best_result \
  --result-boundary-target-temperature 0.25 \
  --result-boundary-target-min-probability-floor 0.0 \
  --result-boundary-target-chunk-size 64 \
  --calculator-result-head-hidden-size 0 \
  --input-proj-lr 0.01 \
  --upstream-lr 0.0003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --snapshot-every 25 \
  --checkpoint-every 25 \
  --snapshot-samples 400 \
  --seed "$SEED" \
  --run-root runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_seed_replication/stage2_seed_${SEED}
```

Retention pass gate for each seed:

```text
final hard learned calculator-result accuracy >= 0.70
final full-enum learned-result best fraction >= 0.70
best post-start target-off checkpoint retains >= 90% of selected Stage 1 hard result accuracy
semantic decoder delta = 0.0
result_boundary_target_loss_weight = 0.0
aux/adaptive/expected/relaxed-entropy/anchor objectives = 0.0
injection-zero and forced-random controls stay low
```

## Required Diagnostics

For every selected Stage 1 and Stage 2 checkpoint, run:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint <checkpoint> --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --output-dir <checkpoint_parent>/canonical_diagnostic
```

and:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint <checkpoint> --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root <checkpoint_parent>/full_enum_diagnostic
```

Report parameter deltas:

```text
Stage 1 step 0 -> selected Stage 1 checkpoint
Stage 2 start -> final and best post-start Stage 2 checkpoint

semantic decoder L2 / max / changed tensors
calculator_hook.result_proj L2 / max / changed tensors
upstream encoder L2 / max / changed tensors
other interface groups L2 / max / changed tensors
```

## Decision Rules

Use these labels:

```text
exact_grid_seed_replication_pass:
  seeds 4 and 5 both pass Stage 1 and Stage 2; with prior seed 2, the retained
  positive has replicated across effective seeds 2/4/5

exact_grid_seed_replication_mixed:
  at least one of seeds 4 or 5 passes Stage 1 and Stage 2, but not both

exact_grid_seed_replication_negative:
  neither seed 4 nor seed 5 passes Stage 1 and Stage 2
```

If `exact_grid_seed_replication_pass`, the next task should be canonical-query
or calculator-protocol stabilization from the retained result-space interface.

If `mixed` or `negative`, the next task should analyze seed fragility and then
compare against multi-sample result-space policy gradient with per-prompt or
leave-one-out baselines. Do not return to oracle checks or frozen-head
boundary-target variants.

## Reporting Requirements

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-exact-grid-retained-positive-seed-replication.md
```

If the task is completed, move this file to:

```text
aiAgentProjectTasks/completed/phase7/
```

Commit and push.

Do not present oracle-at-eval, forced-true result, or decoder-readout checks as
new progress. They are regression checks only. The result that matters is
learned hard calculator-result behavior from the model-side request retained
after the boundary target is exactly off.
