# Phase 6 Third Task: Strict Random-Upstream Local-Target Discovery

## Claim

Phase 6 has now produced the full-model branch result that the overarching
plan required before trying stricter discovery:

```text
With the Stage 0B full-model load, the answer-derived hard-best full-enum local
target can replace direct true-operand labels for Stage 1 teaching, and the
resulting learned calculator-query protocol is retained after the local target
is exactly 0.0.
```

The next best task is therefore not another full-model optimization pass. It is
the stricter branch:

```text
Can the same answer-derived local target teach and retain the true calculator
query when only the frozen semantic decoder is loaded, leaving the upstream
encoder and calculator input projection random/new?
```

This is the fastest route toward the end goal because it directly removes the
largest remaining interpretive crutch: the oracle-trained upstream
representation from Stage 0B.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 4 made the task identifiable with `sum_left_operand`; the answer now
  needs both the calculator result and the left operand.
- Phase 4 showed `operand_spans` exposes enough information for the interface
  head to learn two-digit operands, and teacher-zero answer-only retention can
  hold the true protocol.
- Phase 5 showed answer-only no-handoff training does not discover the
  protocol by itself, but upstream-open continuation can preserve or complete a
  partially taught protocol.
- Phase 6 first showed the local answer-derived full-enum target is sharp:
  best=true `1.000`, tie-aware true-best `1.000`, effective pairs about
  `1.079`, true-pair probability about `0.989`.
- Phase 6 second showed the decisive full-model positive: matched local-target
  teaching reached exact canonical/private/full-enum protocol metrics and
  retained them with local target exactly off.

Less helpful to repeat now:

- Oracle-only runs beyond the strict wiring gate. They validate the decoder
  path but do not answer the learned-interface question.
- More full-model frozen-upstream local-target training. That branch has
  already passed.
- Broad answer-only no-handoff sweeps. Phase 5 already made this a low-value
  direction.
- Gumbel, joint-pair, or soft-target work as the immediate next step. Those are
  useful if strict random-upstream optimization fails, but the sharp hard-best
  target has not yet been tested in the stricter load-scope branch.
- Scale or multi-task broadening. The `0..19` identifiable task still has the
  unresolved strict-random question.

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
aiAgentWorkHistory/phase6/2026-05-10-matched-local-target-teaching-and-retention-gate.md
```

Inspect:

```text
scripts/run_phase6_matched_local_target_teaching.py
scripts/run_phase6_identifiable_full_enum_local_target.py
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
scripts/overfit_one_batch.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_private_protocol.py
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
calculator_action_head=independent_operands
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Use the standard Stage 0B checkpoint as the source for semantic decoder weights:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If absent, use the absolute path recorded in:

```text
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
```

But in this task the load scope must be:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
```

Use a new run root:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target
```

## Part 1: Parameterize The Phase 6 Runner

Extend the Phase 6 matched local-target runner, or create a narrow successor:

```text
scripts/run_phase6_strict_random_upstream_local_target.py
```

The runner should support:

```text
compare-local-target-to-aux
oracle-wiring-gate
run-stage1
run-retention
diagnostics
summarize
```

Required runner changes:

- Add `--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only`.
- Default this new runner to `semantic_decoder_only`.
- Include the load scope in run labels, `commands.jsonl`, `metrics.json`,
  summaries, and work history.
- Make `compare-local-target-to-aux` build/load the model through the same
  code path as training so that the parity gate reflects the actual strict
  branch initialization.
- Keep `full_model` available only as a regression/control option; do not make
  it a primary branch in this task.

Verification:

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase6_strict_random_upstream_local_target.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

If a small focused test is needed, add one around passing
`--semantic-decoder-checkpoint-load-scope semantic_decoder_only` through the
new runner command builder.

## Part 2: Strict Branch Wiring And Target Gates

Before training, run two gates on the semantic-decoder-only initialization.

### Gate A: Oracle-At-Eval Wiring

Purpose:

```text
Verify that the frozen semantic decoder can still answer when the calculator is
fed true/oracle actions, even though the upstream/interface parameters are new.
```

Required evidence:

- oracle-at-eval exact near `1.000`;
- injection-zero exact near `0.0`;
- forced-random near chance;
- semantic decoder parameter delta `0.0`.

If oracle-at-eval fails, stop. The strict branch is not interpretable until the
semantic decoder load-scope or diagnostic wiring is fixed.

### Gate B: Local-Target Parity

Run `compare-local-target-to-aux` on a fixed 128-sample batch under:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
```

Report:

- hard-best pair equals true pair;
- hard-best A/B targets equal true A/B;
- hard-best local CE and direct aux CE on the same logits;
- local-minus-aux CE;
- effective pairs;
- true-pair probability;
- semantic decoder grad/delta after one local step;
- upstream/input-proj delta after that one local step.

Pass gate:

```text
hard_best_pair_equals_true_pair >= 0.98
abs(local_minus_aux_ce) <= 1e-6
semantic_decoder_grad_l2 == 0.0
semantic_decoder_delta_l2 == 0.0
```

If this fails while Gate A passes, stop and debug target construction before
training. Do not interpret a failed parity gate as evidence against discovery.

## Part 3: Stage 1 Strict Local-Target Teaching

This stage may use the answer-derived local target, but must not use direct
true-operand labels.

### Branch A: Frozen Random-Upstream, Input-Projection Only

Purpose:

```text
Can the answer-derived local target train a calculator interface from random
upstream features without any true-operand supervision?
```

Use:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_decay_steps=0
target_mode=hard_best_pair
action_loss_full_enum_temperature=0.25
action_loss_full_enum_chunk_size=64
action_loss_full_enum_min_probability_floor=0.0
input_proj_lr=0.03
upstream_lr=0.003
steps=300
snapshot_every=25
checkpoint_every=25
seed=0
```

Interpretation:

- If this reaches exact or near-exact protocol metrics, the local target can
  train the interface without Stage 0B upstream structure.
- If this partially improves but misses the gate, run Branch B.
- If it is completely flat despite parity passing, inspect whether the random
  upstream representation at `operand_spans` is linearly readable before
  switching estimators.

### Branch B: Upstream-Open Strict Teaching

Run only if Branch A is partial or flat.

Purpose:

```text
Can upstream movement shape the random/new upstream representation under the
answer-derived local target?
```

Use:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=false
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_lr=0.003
upstream_lr=0.0003
steps=500
snapshot_every=25
checkpoint_every=25
seed=0
```

If Branch B is unstable, allow exactly one conservative fallback:

```text
input_proj_lr=0.001
upstream_lr=0.00003
steps=750
```

Do not turn this into a broad LR sweep. The task is to identify whether the
strict branch is immediately viable.

## Part 4: Stage 2 Local-Target-Off Retention

If any Stage 1 snapshot reaches:

```text
fast-gate operand_exact >= 0.90
fast-gate pair_exact >= 0.90
fast-gate calculator_result_accuracy >= 0.90
```

run retention from the first qualifying snapshot and the best qualifying
snapshot.

Use:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint=<selected Stage 1 checkpoint>
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
seed=0
```

For the first retention pass, match the Stage 1 upstream setting:

- if the source was Branch A, use `freeze_upstream_encoder=true`;
- if the source was Branch B, use `freeze_upstream_encoder=false` and
  `upstream_lr=0.00003`.

Retention success requires:

```text
final local_target_loss_weight = 0.0
final aux_operand_loss_weight = 0.0
semantic decoder delta = 0.0
canonical operand/pair/calc near exact
private operand/pair/calc near exact
full-enum learned-minus-true gap near 0.0
full-enum learned-minus-best gap near 0.0
learned-best fraction near 1.0
```

## Required Diagnostics

For every selected checkpoint, run:

```text
python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
python3 scripts/diagnose_private_protocol.py ...
python3 scripts/run_full_enum_action_loss_diagnostic.py ...
```

Selected checkpoints:

- semantic-decoder-only baseline before Stage 1;
- Gate A oracle-wiring checkpoint/summary;
- Gate B parity summary;
- best Stage 1 Branch A checkpoint and final Branch A;
- best Stage 1 Branch B checkpoint and final Branch B, if run;
- first and best Stage 2 retention checkpoints, if retention is run;
- final Stage 2 retention checkpoint.

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
- full-enum learned NLL, true NLL, and best NLL;
- learned-minus-true and learned-minus-best gaps;
- learned-best fraction;
- true-best fraction;
- final aux/local/adaptive/anchor weights;
- trainable parameter groups;
- semantic decoder parameter delta;
- `calculator_hook.input_proj` and upstream deltas;
- semantic decoder checkpoint load scope.

## Decision Criteria

### Strong Strict-Branch Positive

Stage 1 reaches the protocol gate under `semantic_decoder_only`, and Stage 2
retains it with local target exactly off.

This supports:

```text
An answer-derived local interface target can teach and retain the calculator
query protocol without direct true-operand supervision and without inheriting
the oracle-trained upstream representation.
```

This is still local-target-assisted discovery, not pure answer-only discovery.

### Useful Partial Positive

Stage 1 reaches near-exact but Stage 2 retention fails.

Interpretation:

```text
The strict branch can be taught by the local answer-derived target, but the
answer-only retention handoff is unstable without the full-model upstream
structure.
```

Next task should focus on retention schedule/stability, not target sharpness.

### Useful Negative

Gate A and Gate B pass, but Stage 1 cannot train the protocol.

Interpretation:

```text
The target remains sharp, but random/new upstream features or the independent
operand-head optimization path are blocking training.
```

Then the next task should choose one targeted change:

- upstream-open strict teaching with a slower schedule if Branch B was not
  already run;
- a linear-readability probe for random `operand_spans`;
- operand-span joint-pair head with direct full-pair CE;
- Gumbel/Concrete relaxation if optimization appears to be the blocker.

### Stop Conditions

Stop before training if:

- oracle-at-eval does not pass under `semantic_decoder_only`;
- parity local CE no longer matches aux CE on the same logits;
- semantic decoder gradients or parameter deltas are nonzero during the local
  target gate.

Do not run additional oracle-only experiments after the wiring gate passes.

## Required Outputs

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Create:

```text
aiAgentWorkHistory/phase6/2026-05-11-strict-random-upstream-local-target-discovery.md
```

The work history must include:

- code changes;
- exact commands;
- run paths;
- semantic decoder load-scope confirmation;
- oracle-wiring gate;
- parity-gate results;
- Stage 1 teaching table;
- Stage 2 retention table if run;
- selected checkpoint diagnostics;
- final objective weights;
- parameter movement summary;
- comparison to Phase 6 full-model positive;
- go/no-go recommendation.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push.
