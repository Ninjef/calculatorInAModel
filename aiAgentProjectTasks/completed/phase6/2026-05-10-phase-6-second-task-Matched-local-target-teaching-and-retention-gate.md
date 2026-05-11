# Phase 6 Second Task: Matched Local-Target Teaching and Retention Gate

## Claim

Phase 6 first proved the important new fact:

```text
In the identifiable sum_left_operand task, frozen answer-decoder NLL over the
full 20 x 20 action space identifies the true calculator-query pair almost
perfectly.
```

The first smoke also showed that this answer-derived local target is useful,
but the training recipe was intentionally conservative and did not reach the
retention gate:

```text
best upstream-open canonical operand/pair/calc = 0.734
best frozen-upstream canonical operand/pair/calc = 0.566
no snapshot reached >= 0.90
final checkpoints drifted while the local target was still on
```

This task should test the most direct next hypothesis:

```text
If the Phase 6 hard-best local target is as sharp as measured, then using the
same teaching shape that worked in Phase 4 direct-supervision Stage 1 should
teach the true protocol without true-operand labels.
```

In other words, do not change the research question yet. First close the
obvious optimization mismatch between:

- Phase 4 successful direct operand teaching: `answer_loss_weight=0.0`,
  `aux_operand_loss_weight=1.0`, frozen upstream, `input_proj_lr=0.03`.
- Phase 6 first smoke: `answer_loss_weight=1.0`,
  `local_target_loss_weight=1.0`, frozen branch `input_proj_lr=0.001`.

If matched local-target teaching reaches the protocol gate, immediately run
local-target-off retention. That is the fastest path to a strong Phase 6
positive.

## Why This Is The Next Best Task

Helpful prior findings:

- Phase 4 made the task identifiable with `answer_format=sum_left_operand` and
  `calculator_output_format=sum_left_operand`.
- Phase 4 found that `calculator_read_position=operand_spans` exposes enough
  frozen upstream information for `calculator_hook.input_proj` to learn the
  true two-digit operands.
- Phase 4 showed teacher-zero retention can keep a taught protocol after direct
  operand supervision reaches exactly `0.0`.
- Phase 5 showed upstream-open answer-only continuation can complete partially
  taught protocols, but plain no-handoff answer-only discovery is not enough.
- Phase 6 first task showed the answer-derived full-enum target is sharply
  identified in the current setup: best=true `1.000`, tie-aware true-best
  `1.000`, effective pairs about `1.079`, true-pair probability about `0.989`.
- Phase 6 first task also showed the local target improves protocol metrics
  materially above Phase 5 no-handoff answer-only training.

Less helpful to repeat now:

- Oracle-only runs. They are wiring checks, not progress.
- Broad answer-only no-handoff sweeps. Phase 5 already answered that.
- Addition-only full-enum/action-loss experiments. They were underidentified
  and are not the Phase 6 setting.
- Sampled-candidate replay/EMA in the full `20 x 20` action space. Full enum is
  available and already diagnostic.
- Strict random-upstream discovery. It should wait until the full-model branch
  passes.
- Joint-pair/Gumbel work as the immediate next step. Those may be useful, but
  the first smoke has not yet tested the direct Phase 4-matched recipe with the
  now-sharp local target.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-10-phase-6-overarching_plan-Identifiable-local-interface-discovery.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-10-identifiable-full-enum-local-target-sharpness-and-smoke.md
```

Inspect:

```text
scripts/run_phase6_identifiable_full_enum_local_target.py
scripts/overfit_one_batch.py
scripts/run_full_enum_action_loss_diagnostic.py
src/model.py
tests/test_model.py
```

## Fixed Setup

Keep the Phase 4/5/6 identifiable setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
answer_format=sum_left_operand
calculator_output_format=sum_left_operand
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
calculator_estimator=identifiable_full_enum_local_target for Stage 1
calculator_action_head=independent_operands
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Use the standard Stage 0B checkpoint:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If absent, use the absolute path recorded in the Phase 4 fact sheet.

Use a new run root:

```text
runs/2026-05-10_phase6_matched_local_target_teaching
```

## Part 1: Add A Parity Gate

Before running the ladder, add a small parity diagnostic to the Phase 6 runner
or to a narrowly named helper:

```text
compare-local-target-to-aux
```

For a fixed batch from the Stage 0B full-model load, report:

- hard-best full-enum pair equals true operand pair;
- hard-best A/B targets equal true A/B targets;
- hard-best local CE and direct aux CE on the same logits;
- local target metrics: best=true, true-pair rank, target entropy, effective
  pairs, true-pair probability;
- no gradient into the semantic decoder;
- no use of true operands in constructing the local target.

Pass gate:

```text
hard_best_pair_equals_true_pair >= 0.98
local_target_ce approximately equals aux_ce on the same logits
semantic_decoder_delta = 0.0
```

If this fails, stop and debug target construction. Do not proceed to training.

## Part 2: Extend The Runner For Matched Teaching

Extend `scripts/run_phase6_identifiable_full_enum_local_target.py` or create:

```text
scripts/run_phase6_matched_local_target_teaching.py
```

The runner should support:

```text
compare-local-target-to-aux
run-stage1
run-retention
diagnostics
summarize
```

Add runner flags instead of hardcoding the first-smoke recipe:

```text
--answer-loss-weight
--local-target-loss-weight
--input-proj-lr
--steps
--snapshot-every
--checkpoint-every
--target-mode hard_best_pair | soft_pair
```

For this task, use:

```text
target_mode=hard_best_pair
action_loss_full_enum_temperature=0.25
action_loss_full_enum_chunk_size=64
action_loss_full_enum_min_probability_floor=0.0
```

Keep exact commands in:

```text
runs/2026-05-10_phase6_matched_local_target_teaching/commands.jsonl
```

## Part 3: Stage 1 Local-Target Teaching

This stage is allowed to use the answer-derived local target, but must not use
direct true-operand labels.

### Branch A: Phase 4-Matched Frozen-Upstream Teaching

Purpose:

```text
Can the answer-derived hard-best local target replace Phase 4's direct operand
aux labels when the rest of the teaching recipe is matched?
```

Use:

```text
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_decay_steps=0
input_proj_lr=0.03
upstream_lr=0.003
steps=300
snapshot_every=25
checkpoint_every=25
seed=0
```

Interpretation:

- If this reaches exact or near-exact protocol metrics, the Phase 6 local target
  can replace true operand supervision for Stage 1 teaching in the full-model
  branch.
- If it overshoots or oscillates, run Branch B.
- If it stays partial despite the parity gate passing, the blocker is probably
  not target identifiability; inspect gradients/loss scale before changing
  architecture.

### Branch B: Conservative Matched Teaching Fallback

Run only if Branch A overshoots, diverges, or peaks below the retention gate.

Use:

```text
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_lr=0.01
upstream_lr=0.003
steps=500
snapshot_every=25
checkpoint_every=25
seed=0
```

Do not add more LR variants in this task unless one branch is within a few
points of the gate and a single immediate continuation is clearly justified.

### Optional Branch C: Local-Target Weight Scale Check

Run only if both A and B are stable but underfit.

Use:

```text
freeze_upstream_encoder=true
answer_loss_weight=0.0
local_target_loss_weight=3.0
input_proj_lr=0.003
steps=500
snapshot_every=25
checkpoint_every=25
```

This checks whether the first smoke was simply too weak without making a broad
sweep.

## Part 4: Stage 2 Local-Target-Off Retention

If any Stage 1 snapshot reaches:

```text
fast-gate operand_exact >= 0.90
fast-gate pair_exact >= 0.90
fast-gate calculator_result_accuracy >= 0.90
```

start retention from the first and best qualifying snapshots.

Use:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint=<selected Stage 1 checkpoint>
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
upstream_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
```

This is the critical Phase 6 claim. Stage 1 alone is not enough.

Success requires the selected Stage 2 checkpoint to keep:

```text
canonical operand/pair/calc near exact
private operand/pair/calc near exact
full-enum learned-minus-true gap near 0.0
full-enum learned-minus-best gap near 0.0
learned-best fraction near 1.0
semantic decoder delta = 0.0
final local_target_loss_weight = 0.0
final aux_operand_loss_weight = 0.0
```

## Part 5: Optional Upstream-Open Retention

Run only if frozen-upstream Stage 2 retention succeeds.

Purpose:

```text
Can answer-only continuation retain the locally discovered protocol while
upstream parameters are allowed to move conservatively?
```

Use:

```text
freeze_upstream_encoder=false
answer_loss_weight=1.0
local_target_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_lr=0.0003
upstream_lr=0.00003
steps=1000
snapshot_every=50
checkpoint_every=50
```

This is useful but secondary. Do not let it delay the frozen-upstream retention
diagnostics.

## Required Diagnostics

For each selected checkpoint, run:

```text
python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
python3 scripts/diagnose_private_protocol.py ...
python3 scripts/run_full_enum_action_loss_diagnostic.py ...
```

Selected checkpoints:

- Stage 0B baseline;
- parity-gate batch summary;
- best Stage 1 local-target teaching checkpoint;
- final Stage 1 checkpoint;
- first Stage 2 retention checkpoint crossing the gate, if any;
- best Stage 2 retention checkpoint, if any;
- final Stage 2 retention checkpoint;
- optional upstream-open retention best/final checkpoints, if run.

Report at minimum:

- built-in eval exact;
- normal exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- oracle-at-eval exact;
- learned operand exact;
- learned pair exact;
- learned calculator-result accuracy;
- private all-pair answer exact;
- private all-pair operand/pair/calc;
- full-enum learned NLL, true NLL, best NLL;
- learned-minus-true and learned-minus-best gaps;
- learned-best fraction;
- true-best fraction;
- final aux/local/adaptive/anchor weights;
- trainable parameter groups;
- semantic decoder parameter delta;
- `calculator_hook.input_proj` and upstream deltas.

## Decision Criteria

### Strong Positive

Stage 1 reaches the protocol gate without direct operand supervision, and Stage
2 retains it with local target exactly off.

This would support:

```text
In the full-model Phase 4/5 identifiable branch, an answer-derived local
interface target can teach the calculator-query protocol without direct
true-operand supervision, and answer-only training can retain it.
```

This is not strict random-upstream discovery yet.

### Useful Partial Positive

Stage 1 reaches near-exact but Stage 2 retention fails.

Interpretation:

```text
The local target can replace direct operand labels for teaching, but answer-only
retention still needs a better handoff schedule, slower LR, or stability
regularization.
```

Next task should focus on local-target decay or retention stability, not target
construction.

### Useful Negative

The parity gate passes but matched Stage 1 still fails to exceed the first
smoke.

Interpretation:

```text
The target is sharp, but the current independent-head optimization path is
blocked for reasons other than the conservative first-smoke settings.
```

Then the next task should move to one parameterization/estimator change:

- operand-span joint-pair head with direct full-pair CE; or
- Gumbel/Concrete relaxation if the evidence points to answer-loss/retention
  gradient routing as the blocker.

### Stop Condition

Do not proceed to strict `semantic_decoder_only` random-upstream discovery in
this task. That branch only becomes appropriate after the full-model branch has
a retained local-target-off checkpoint.

## Required Outputs

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Create:

```text
aiAgentWorkHistory/phase6/2026-05-10-matched-local-target-teaching-and-retention-gate.md
```

The work history must include:

- code changes;
- exact commands;
- run paths;
- parity-gate results;
- Stage 1 table;
- Stage 2 retention table if run;
- selected checkpoint diagnostics;
- final objective weights;
- parameter movement summary;
- comparison to Phase 4 direct-supervision teaching;
- comparison to Phase 6 first-smoke local-target results;
- go/no-go recommendation.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push.
