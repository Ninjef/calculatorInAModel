# Phase 6 Overarching Plan: Identifiable Local Interface Discovery

## Mission

Phase 6 should move the project from protocol teaching/retention toward
protocol discovery.

Phase 4 established the first clean learned-interface positive:

```text
With an identifiable answer target and a frozen readable upstream
representation, direct operand supervision can teach the true calculator-query
protocol, and answer loss can retain or complete that protocol after direct
teacher weights are exactly 0.0.
```

Phase 5 established the strongest current upstream result:

```text
Upstream movement can preserve and complete an already partially taught
calculator-query protocol, but plain answer-only training does not discover the
protocol without a supervised handoff or local teaching signal.
```

Phase 6 should therefore ask:

```text
Can an answer-derived local interface target, in the identifiable Phase 4/5
task, teach the upstream/model-side calculator-query protocol without using
direct true-operand supervision?
```

This is not a retreat from discovery. It is the smallest honest step toward
discovering a non-differentiable tool-call protocol: use the frozen downstream
answer loss to construct a local training signal at the calculator interface,
then require that learned protocol to survive after the local signal is removed.

## Read First

Before doing Phase 6 work, read these files in this order:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/completed/phase5/2026-05-09-phase-5-closure-Upstream-discovery-after-protocol-teaching.md
```

For local/action-loss history, also read:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_3_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/completed/phase2/2026-05-05-phase-2-ninth-task-Full-action-enumeration-teacher-before-upstream-unfreezing.md
aiAgentProjectTasks/completed/phase3/2026-05-06-phase-3-first-task-Joint-pair-action-interface.md
```

For the current implementation surface, inspect:

```text
src/data.py
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
```

## Critical Guardrail

Do not rediscover oracle success.

Oracle operands, oracle-at-eval, forced true result classes, injection-zero, and
forced-random controls are wiring and bottleneck checks only. They are necessary
controls, but they are not research progress on the central question.

The central Phase 6 evidence must be learned-interface behavior:

- learned operand exact;
- learned pair exact;
- learned calculator-result accuracy;
- private all-pair protocol decoding;
- full-enum learned-minus-true and learned-minus-best action-loss gaps;
- local target / interface-discovery weight exactly `0.0` for retention claims;
- aux/direct true-operand supervision weight exactly `0.0`;
- semantic decoder movement exactly `0.0`;
- dense checkpoint selection whenever upstream or interface parameters move.

## What Phase 6 Is Not

Phase 6 should not be another broad seed/LR sweep of Phase 5 answer-only
training. Phase 5 already gave the answer: plain answer-only no-handoff
training produced only partial transient alignment and final drift toward
chance learned actions.

Phase 6 should not spend primary effort on:

- more oracle-only semantic decoder runs;
- more addition-only action-loss experiments;
- more single-sample REINFORCE variants;
- broad STE sweeps;
- sampled-candidate replay/EMA in the `20 x 20` action space;
- upstream unfreezing without dense checkpoints and protocol diagnostics;
- success claims based on answer exact alone.

## Key Interpretation From Prior Phases

Phase 2 and Phase 3 already tried local/action-loss/full-enum ideas, but mostly
in the older addition-only environment. Those results remain important:

- answer-NLL action landscapes contained useful signal;
- full enumeration was better as a diagnostic than sampled candidates;
- soft targets were often broad because many pairs share the same sum;
- learned-best action-loss fraction stayed `0.0`;
- private-code and underidentification remained central problems.

The Phase 6 bet is not "try full-enum again because nobody tried it." The bet
is sharper:

```text
Try answer-derived local interface targets in the Phase 4/5 identifiable setup,
where the answer requires both the sum and the left operand, so the best
calculator action should identify the intended true query much more sharply.
```

In plain addition, `(3, 7)`, `(4, 6)`, and `(5, 5)` can be equivalent for the
answer. In `sum_left_operand`, those are not equivalent, because the calculator
output includes the learned left operand and the answer decoder must emit it.

## Fixed Phase 6 Baseline Setup

Unless a task explicitly says otherwise, keep the Phase 4/5 setup fixed:

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
freeze_semantic_decoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0 unless explicitly testing a local objective
input_proj_anchor_weight=0.0 unless explicitly testing an anchor control
oracle_train=false
oracle_warmup_steps=0
```

The standard Stage 0B semantic decoder checkpoint is the operand-aware oracle
checkpoint from Phase 4:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If that repo-local path is absent in a sandbox, use the absolute path recorded
in `factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md`.

When using the Stage 0B checkpoint, be explicit about load scope:

- `full_model` reproduces Phase 5 no-handoff full-model initialization.
- `semantic_decoder_only` is the stricter random-upstream branch and should be
  attempted only after the full-model branch produces a meaningful local-target
  discovery result.

## Primary Research Tracks

### Track A: Identifiable Full-Enum Local Target

This is the mainline.

Construct local calculator-interface targets from frozen answer-decoder NLL
over the full `20 x 20` action space in the Phase 4/5 identifiable task.

The local target may use:

- forced calculator action pairs;
- frozen answer loss under each forced action;
- the best or soft-best action distribution derived from those losses.

The local target must not use:

- true operand labels as the training target;
- true sums as the training target;
- oracle operands during training;
- direct `aux_operand_loss`.

First prove the target is sharp. Then train the interface/upstream toward it.
Then turn the local target off exactly and test answer-only retention.

### Track B: Gumbel/Concrete Relaxation

This is the second-line track if Track A shows the local answer-derived target
is informative but hard argmax training remains unstable.

A useful Gumbel task should keep the same identifiable setup and compare:

- hard forward / soft backward through operand distributions;
- temperature schedule;
- final hard-action protocol metrics;
- retention after any relaxation-specific loss/schedule reaches its final
  claim state.

Do not start here unless Track A fails in a way that suggests optimization,
not target identifiability, is the blocker.

### Track C: Strict Random-Upstream Discovery

This is not the first Phase 6 task.

Only attempt strict random-upstream discovery after a local-target method passes
the no-handoff full-model branch. Use:

```text
--semantic-decoder-checkpoint-load-scope semantic_decoder_only
```

and require oracle-at-eval to pass as a wiring gate before interpreting learned
interface failure.

### Track D: Scale or Multi-Task Generalization

Do not broaden to larger operand ranges, more digits, or multi-operation tasks
until Phase 6 has a clear local-target discovery result in `0..19`.

## Phase 6 Standard Stages

### Stage 0: Target Sharpness Diagnostic

Before training a new objective, prove that the local answer-derived target is
actually better identified in the Phase 4/5 task than it was in Phase 2/3.

For a sample of prompts, full-enumerate all `20 x 20` actions and report:

- best action pair;
- whether best pair equals true operands;
- tie-aware true-best fraction;
- best result matches true sum;
- full soft-target entropy / effective pairs;
- true-pair probability under soft target;
- top-k target mass;
- learned action NLL if starting from a checkpoint;
- true action NLL, for reporting only.

Do not train until this diagnostic says the target is sharp enough to be worth
training.

Useful gates:

```text
best-matches-true-operands >= 0.90
tie-aware true-best high under reasonable tolerance
effective pairs far below the old Phase 2/3 value around 29
true-pair soft probability materially above the old tiny marginal masses
```

If the target is not sharp even under `sum_left_operand`, stop and redesign the
answer-derived local target before training.

### Stage 1: No-Handoff Full-Model Local-Target Training

Start from Stage 0B with `semantic_decoder_checkpoint_load_scope=full_model`.

Train without direct operand supervision:

```text
aux_operand_loss_weight=0.0
oracle_train=false
```

The local target objective is allowed to be nonzero in Stage 1, because the
claim is local answer-derived discovery, not answer-only discovery.

Primary trainable options, in order:

1. `calculator_hook.input_proj` only, upstream frozen.
2. `calculator_hook.input_proj` plus upstream, with conservative upstream LR.

The first option tests whether the existing Stage 0B upstream representation is
already linearly readable by an answer-derived local objective. The second
tests whether upstream can discover/shape the representation under that local
objective.

Use dense snapshots every `25` or `50` steps and save checkpoints.

### Stage 2: Local-Target-Off Retention

From selected Stage 1 checkpoints, continue with:

```text
answer_loss_weight=1.0
local_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
```

This is the critical retention claim. Do not call Stage 1 alone a solved result.

### Stage 3: Strict Random-Upstream Branch

Only after Stage 1/2 succeed under full-model load, rerun the same method with:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
```

This is the first branch that can support a stricter no-handoff/random-upstream
discovery claim, assuming oracle-at-eval passes.

## Required Diagnostics

For every selected checkpoint in Phase 6, run:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

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
- trainable parameter groups;
- semantic decoder parameter delta;
- upstream/input-proj parameter deltas when relevant;
- exact final weights for aux, local target, adaptive interface, and anchors.

## Success Definitions

### Useful Stage 0 Positive

The full-enum answer-derived target is sharp in the identifiable setup:

```text
best action usually equals true operands, target entropy is much lower than in
Phase 2/3 addition-only full-enum targets, and true-pair mass is high enough to
train against.
```

### Useful Stage 1 Positive

With direct operand supervision exactly absent, the local answer-derived target
trains learned actions toward the true calculator-query protocol.

This requires protocol metrics, not answer exact alone.

### Strong Phase 6 Positive

After the local target is set exactly to `0.0`, answer-only continuation retains
the true protocol:

```text
canonical operand/pair/calc near exact
private operand/pair/calc near exact
full-enum learned-minus-true and learned-minus-best gaps 0.0 or near 0.0
learned-best fraction near 1.0
semantic decoder delta 0.0
aux/direct operand supervision 0.0
```

### Very Strong Phase 6 Positive

The above result survives the strict `semantic_decoder_only` branch with
random/new upstream parameters, while oracle-at-eval remains a healthy wiring
gate.

## Failure Interpretation

If Stage 0 target sharpness fails:

```text
The identifiable answer setup is still not creating a sharp enough local
answer-derived target, or the full-enum target construction is wrong.
```

Do not proceed to broad training.

If Stage 0 is sharp but Stage 1 training fails with upstream frozen:

```text
The target is informative, but the frozen Stage 0B read representation or
independent-head parameterization is blocking optimization.
```

Next try upstream-open local-target training, or a Gumbel/soft relaxation.

If Stage 1 succeeds but Stage 2 retention fails:

```text
The local target can teach the interface, but answer loss alone still cannot
hold it. Investigate slower local-target decay, shorter handoff windows, or
stability regularization.
```

If full-model load succeeds but semantic-decoder-only fails:

```text
The method can use existing oracle-trained upstream structure, but has not yet
solved strict random-upstream discovery.
```

That is still useful progress; label it honestly.

## Reporting Contract

Every Phase 6 task completion should update or create:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/<date>-<short-description>.md
```

Every completed task should include:

- claim tested;
- code changes;
- exact commands;
- run paths;
- selected checkpoints;
- target sharpness summary if relevant;
- fast-gate table;
- full diagnostics table;
- exact final objective weights;
- parameter movement summary;
- comparison to Phase 4/5 baselines;
- go/no-go recommendation.

When the task is complete, move the task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

then commit and push, following `CLAUDE.md`.

## First Task Recommendation

The first Phase 6 task should be:

```text
Phase 6 First Task: Identifiable Full-Enum Local-Target Sharpness and Smoke
```

It should:

1. Add a Phase 6 fact sheet and work-history folder if missing.
2. Add or adapt a full-enum target sharpness diagnostic for
   `answer_format=sum_left_operand`.
3. Prove whether the full-enum best action now identifies true operands.
4. If and only if target sharpness passes, run a compact no-handoff local-target
   smoke from Stage 0B full-model load.
5. Save dense snapshots and run full diagnostics only on selected checkpoints.

