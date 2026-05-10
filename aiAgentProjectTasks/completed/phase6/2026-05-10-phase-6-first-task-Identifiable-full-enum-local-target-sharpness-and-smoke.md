# Phase 6 First Task: Identifiable Full-Enum Local-Target Sharpness and Smoke

## Claim

Prior full-enum/action-loss training was mostly tested in the older
addition-only setting, where the answer underidentified the true operand pair.

This task tests the Phase 6 bet:

```text
In the Phase 4/5 identifiable setup, answer-derived full-enum local targets
should be much sharper, because the answer requires both the sum and the left
operand. If the target is sharp, it may provide a useful local discovery signal
for the calculator-query interface without direct true-operand supervision.
```

The task has two parts:

1. Target sharpness diagnostic.
2. A compact local-target training smoke, only if the diagnostic passes.

Do not skip Part 1. Do not broaden into seed/LR sweeps in this task.

## Read First

Read:

```text
CLAUDE.md
aiAgentProjectTasks/2026-05-10-phase-6-overarching_plan-Identifiable-local-interface-discovery.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/completed/phase5/2026-05-09-phase-5-closure-Upstream-discovery-after-protocol-teaching.md
aiAgentProjectTasks/completed/phase2/2026-05-05-phase-2-ninth-task-Full-action-enumeration-teacher-before-upstream-unfreezing.md
docs/canonical_diagnostics.md
```

Inspect:

```text
scripts/overfit_one_batch.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
src/model.py
src/data.py
```

## Fixed Setup

Use the Phase 4/5 identifiable task:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
answer_format=sum_left_operand
calculator_output_format=sum_left_operand
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Stage 0B checkpoint:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If absent, use the absolute Stage 0B path recorded in:

```text
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
```

Use a new run root:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target
```

Prefer a reusable runner:

```text
scripts/run_phase6_identifiable_full_enum_local_target.py
```

The runner should support:

```text
summarize-target
run-smoke
diagnostics
summarize
```

If a runner is too much for the first pass, keep commands exact and reproducible
in work history, but do not leave the result as an undocumented ad hoc run.

## Part 0: Phase 6 Scaffolding

Create if missing:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/
aiAgentProjectTasks/completed/phase6/
```

The Phase 6 fact sheet should start with:

- the mission from the overarching plan;
- the Phase 4/5 fixed setup;
- the anti-oracle guardrail;
- an empty section for this task's results.

## Part 1: Target Sharpness Diagnostic

Add or adapt a diagnostic that full-enumerates all `20 x 20` action pairs under
the identifiable setup and reports target sharpness before training.

This can extend `scripts/run_full_enum_action_loss_diagnostic.py` or be a small
Phase 6 runner/helper that reuses its scoring utilities.

For each prompt:

1. Enumerate every pair `(a, b)` in `[0, 19] x [0, 19]`.
2. Force each pair through the real calculator/output path.
3. Compute frozen answer-decoder NLL for the target answer.
4. Rank all pairs by NLL.
5. Convert losses to soft weights using the existing temperature machinery.

The training target construction may use answer NLL only. It must not use true
operands or true sums.

The diagnostic may report true operands for evaluation:

- best pair equals true pair;
- tie-aware true-best fraction;
- true pair rank;
- true pair NLL;
- best pair NLL;
- best pair result accuracy;
- best pair left-operand accuracy;
- soft target entropy;
- soft target effective pairs;
- true pair soft probability;
- top-1, top-3, top-5 mass;
- true A/B marginal mass;
- comparison to Phase 2/3 broad target facts where useful.

Run the diagnostic on at least:

1. Stage 0B full-model load with the learned interface untrained/random.
2. A known Phase 4 retained true-protocol checkpoint, as a positive sanity
   check if readily available.
3. Optionally, a Phase 5 no-handoff best partial checkpoint for comparison.

Suggested sample sizes:

```text
samples=128 for quick target development
samples=400 full private all-pair sweep if runtime is reasonable
```

Target-sharpness pass gate:

```text
best_pair_equals_true_pair >= 0.90
true_pair_rank close to 1.0 average
soft target effective pairs much lower than the old Phase 2/3 addition-only
value around 29
true_pair soft probability materially higher than old broad marginal masses
```

If this gate fails, stop. Write the negative result and do not train the smoke.

## Part 2: Implement the Local-Target Smoke Objective

Only if Part 1 passes, implement a minimal objective for the identifiable setup.

Prefer a narrowly named estimator:

```text
calculator_estimator=identifiable_full_enum_local_target
```

or, if less invasive, add a mode/flag to reuse
`action_loss_full_enum_interface` while making the answer format and diagnostic
contract Phase 6 explicit.

Required behavior:

- full-enumerate all `20 x 20` forced action pairs for each batch prompt;
- compute answer NLL for every pair with the semantic decoder frozen;
- construct a local target from answer NLL only;
- train the learned interface toward that target;
- do not use `aux_operand_loss`;
- do not use true operands/sums to construct the target;
- log target sharpness metrics during training.

Start with independent operand heads and `operand_spans`; do not switch to the
old `joint_pair` head in this first task unless independent heads clearly block
implementation.

Recommended local target variants, in order:

1. Hard best-pair CE against the single best answer-NLL pair.
2. Soft full-pair distribution, marginalized to A/B, using a low temperature.

If both are easy, support both behind a flag, but run only the hard best-pair CE
first. The central question is whether the identifiable answer makes the best
pair equal the true protocol.

Suggested flags:

```text
--local-target-loss-weight
--local-target-decay-steps
--local-target-temperature
--local-target-mode hard_best_pair | soft_pair
--local-target-full-enum-chunk-size
```

Record final local-target weight in `metrics.json`.

## Part 3: Stage 1 Smoke Training

Run at most two compact smoke branches.

### Branch A: Frozen Upstream, Input-Projection Only

Purpose:

```text
Can answer-derived local targets train the calculator interface using the
existing Stage 0B upstream representation, without direct operand supervision?
```

Use:

```text
semantic_decoder_checkpoint_load_scope=full_model
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_lr=0.0003 or 0.001
steps=500
snapshot_every=25 or 50
checkpoint_every=25 or 50
```

One seed is enough for this first smoke. Prefer CLI seed `0`.

### Branch B: Upstream Open, Conservative LR

Run only if Branch A target sharpness is good but training is partial or
clearly underfits.

Purpose:

```text
Can upstream movement plus the local answer-derived target discover/shape the
calculator read representation?
```

Use:

```text
freeze_upstream_encoder=false
input_proj_lr=0.0003
upstream_lr=0.00003
steps=1000
snapshot_every=50
checkpoint_every=50
```

Again, one seed is enough unless a near-positive needs immediate confirmation.

Do not run more than Branch A plus Branch B in this task.

## Part 4: Stage 2 Local-Target-Off Retention

If any Stage 1 snapshot reaches:

```text
fast-gate operand/pair/calc >= 0.90
```

start a retention continuation from the best snapshot:

```text
answer_loss_weight=1.0
local_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
```

Use:

```text
steps=500 or 1000
snapshot_every=50
checkpoint_every=50
```

This is the key claim checkpoint. Stage 1 local-target training alone is not
enough to claim retained calculator use.

## Part 5: Required Diagnostics

For each selected checkpoint, run:

```text
python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
python3 scripts/diagnose_private_protocol.py ...
python3 scripts/run_full_enum_action_loss_diagnostic.py ...
```

Selected checkpoints:

- Stage 0 target-sharpness checkpoint(s);
- best Stage 1 smoke checkpoint if any fast-gate operand/pair/calc exceeds
  `0.35`;
- final Stage 1 checkpoint;
- best Stage 2 retention checkpoint if Stage 2 is run;
- final Stage 2 checkpoint if Stage 2 is run.

Report:

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
- full-enum learned-minus-true gap;
- full-enum learned-minus-best gap;
- learned-best fraction;
- true-best fraction;
- semantic decoder parameter delta;
- input-proj and upstream parameter deltas;
- final aux weight;
- final local-target weight;
- trainable parameter groups.

## Decision Criteria

### Stop After Part 1 If

The identifiable full-enum answer-derived target is still broad:

```text
best_pair_equals_true_pair < 0.90
or target effective pairs remains close to old addition-only values
```

Write the negative result and recommend target redesign instead of training.

### Useful Positive

Stage 1 local-target training improves learned protocol metrics without direct
operand supervision:

```text
canonical/private operand or pair exact materially exceeds the Phase 5
no-handoff best partial checkpoints
```

with:

```text
aux_operand_loss_weight = 0.0
oracle_train = false
semantic decoder delta = 0.0
```

### Strong Positive

Stage 2 retention keeps a near-true protocol after the local target is exactly
off:

```text
local_target_loss_weight = 0.0
aux_operand_loss_weight = 0.0
canonical operand/pair/calc near exact
private operand/pair/calc near exact
full-enum learned-minus-true/best gaps near 0.0
learned-best fraction near 1.0
```

### Negative But Useful

If target sharpness is strong but training fails, the failure is likely
optimization or parameterization. Recommend one of:

- stronger hard-best target weight;
- lower/higher input-proj LR;
- upstream-open local target;
- Gumbel/Concrete relaxation;
- joint-pair head adapted to `operand_spans`.

Do not recommend broad answer-only sweeps.

## Required Outputs

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Create:

```text
aiAgentWorkHistory/phase6/2026-05-10-identifiable-full-enum-local-target-sharpness-and-smoke.md
```

The work history must include:

- code changes;
- exact commands;
- run paths;
- target sharpness table;
- smoke/retention table if run;
- selected checkpoint diagnostics;
- exact final weights for aux/local/adaptive/anchor objectives;
- parameter movement summary;
- comparison to Phase 5 no-handoff smoke;
- next-step recommendation.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push.

