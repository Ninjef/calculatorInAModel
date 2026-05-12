# Phase 6 Ninth Task: Sum-Only Semantic Decoder Gate And Natural Bridge Readiness

## Mission

Unblock the natural sum-only relaxed bridge without confusing decoder wiring
work for learned-interface progress.

The latest Phase 6 result says the deterministic hard-forward /
soft-backward Concrete bridge is the strongest current method in the
identifiable `sum_left_operand` setup, but the first natural sum-only attempt
stopped at Stage 0 because the downstream semantic decoder was not healthy
enough:

```text
oracle-at-eval exact stayed around 0.91-0.95
full-enum best-result group matched the true sum only about 0.906
```

This task should answer:

```text
Can we produce a strict natural sum-only semantic decoder/wiring gate that is
good enough to make deterministic relaxed-bridge training interpretable?
```

If and only if that gate passes, run the smallest natural sum-only
deterministic relaxed bridge and retention check.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 4/5 made the calculator-query protocol identifiable with
  `sum_left_operand` and showed taught protocols can be retained after direct
  supervision is removed.
- Phase 6 full-enum hard-best targets showed the frozen answer decoder creates
  an extremely sharp action landscape in the identifiable task.
- Phase 6 strict random-upstream local-target training solved the interface
  with only the frozen semantic decoder loaded, then retained exactly with
  local target weight `0.0`.
- Phase 6 deterministic hard-forward / soft-backward Concrete training is the
  most end-goal-relevant positive so far: answer loss trained hard calculator
  actions without true operand labels, hard-best CE, expected-answer-loss
  objectives, or oracle operands during training.
- The deterministic relaxed bridge replicated across effective seeds `2`, `4`,
  and `5`, and relaxation-off answer-only continuation retained or completed
  exact protocols.
- A carefully upstream-open deterministic branch also retained exactly, so
  the method is not obviously limited to a brittle frozen-interface trick.

Less helpful directions right now:

- More oracle-only success in already healthy `sum_left_operand` wiring.
  Oracle-at-eval is a gate, not progress.
- More hard-best local-target teaching in the identifiable task. It already
  works and is now mainly a control.
- More simple linear local-target decay. The tested strict decay ladder through
  `150` steps failed to hand off cleanly.
- More independent-head exact expected answer-loss sweeps. Expected loss fell,
  but hard argmax actions collapsed to the wrong protocol.
- More literal stochastic Gumbel before its instability is fixed. The one-step
  gate was positive, but training stayed near chance and reached `NaN`.
- Scaling to `operand_max=99` from a blocked natural `0..19` decoder. That
  would make a wiring problem harder to diagnose.

The fastest useful path is therefore:

```text
make natural sum-only Stage 0 interpretable -> rerun deterministic Concrete
only if the gate passes -> report result-level learned calculator use.
```

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
aiAgentWorkHistory/phase6/2026-05-12-relaxed-bridge-replication-stochastic-upstream.md
aiAgentWorkHistory/phase6/2026-05-12-natural-sum-only-relaxed-bridge.md
```

Inspect:

```text
src/data.py
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_natural_sum_only_relaxed_bridge.py
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Fixed Natural Setup

Unless a candidate branch explicitly says otherwise:

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
calculator_action_head=independent_operands
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Use the previous natural runner/run root as a reference, but write new outputs
under a new root:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate
```

The previous natural run root remains important context:

```text
runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge
```

## Critical Guardrail

This is partly a wiring task. Treat it honestly.

Allowed during Stage 0 decoder/wiring work:

- oracle training of a sum-only semantic decoder;
- architecture/read-position comparisons needed to make the answer decoder
  reliable;
- true sums and true operands for diagnostics only;
- full-enum result-aware diagnostics.

Forbidden as a progress claim:

- presenting oracle-trained decoder exactness as learned calculator use;
- counting oracle-at-eval recovery as Phase 6 success;
- running bridge training when oracle-at-eval or full-enum result matching is
  below gate;
- interpreting pair-exact failure as failure in natural sum-only branches when
  the learned calculator result is correct.

Allowed during bridge training only after Stage 0 passes:

- `calculator_estimator=gumbel_concrete_interface`;
- deterministic Concrete/softmax relaxation;
- hard-forward / soft-backward calculator signals;
- answer loss only;
- optional relaxation-off retention with `calculator_estimator=adaptive_interface`.

Forbidden during bridge training:

- true operand CE;
- hard-best pair CE;
- full-enum soft target CE;
- `calculator_estimator=identifiable_full_enum_local_target`;
- `calculator_estimator=full_enum_expected_answer_loss`;
- oracle operands;
- semantic decoder movement.

## Stage 0A: Diagnose The Existing Natural Failure

Start from the existing natural runner and artifacts. Reproduce or load the
latest summary, then add enough diagnostics to explain the decoder failure.

Required questions:

- Are the oracle misses concentrated on particular sums, token positions, or
  carry boundaries?
- Does `oracle-at-eval` fail because the semantic decoder cannot decode the
  calculator sum, or because the read-position/checkpoint shape is mismatched?
- Does forced true result class agree with oracle operand injection?
- Does the existing April sum-only checkpoint metadata match the current
  `operand_spans` / `answer_decoder` / `sum` assumptions?
- Is the full-enum best-result mismatch caused by true-sum decoding errors or
  by broad same-sum underidentification?

Use the all-400-example space when it is cheap and helpful. The natural
`0..19` task is small enough that aggregate eval samples can hide systematic
misses.

Deliverable:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate/stage0_existing_decoder_diagnosis.json
```

and a compact Markdown summary.

## Stage 0B: Produce A Passing Sum-Only Gate

If Stage 0A reveals a code/config bug, fix it narrowly and rerun the gate.

If the issue appears to be decoder capacity or read-position compatibility,
run a small candidate ladder. Keep it narrow and stop at the first passing
candidate.

Suggested candidate order:

1. Current tiny setup, but select dense oracle checkpoints by the Stage 0 gate
   metrics instead of final training loss.
2. Same setup with `calculator_read_position=operands` if the metadata audit
   suggests the older sum-only decoder expected that path.
3. Slightly stronger decoder/model shape, such as `n_embd=32`, `n_head=2`,
   `n_layer=2`, keeping `operand_max=19` and `calculator_output_format=sum`.
4. Only if needed, `n_layer=3` with the same `n_embd=32`, `n_head=2` shape.

Do not run a broad sweep. The goal is to find whether natural sum-only is
currently blocked by simple wiring/capacity, not to optimize oracle training
as an end in itself.

For every candidate, report:

- exact command;
- checkpoint selected;
- built-in eval exact;
- oracle-at-eval exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- full-enum best-result group matches true sum;
- full-enum learned-result metrics at initialization when applicable;
- semantic decoder delta during later frozen checks.

Passing gate:

```text
oracle-at-eval exact >= 0.98
full-enum best-result group matches true sum >= 0.98
injection-zero and forced-random remain near chance
semantic decoder delta == 0.0 in frozen evaluation/bridge checks
```

If no candidate passes, stop after Stage 0B. Update the fact sheet and work
history with a clear blocker diagnosis and next recommendation. Do not run
Stage 1 bridge training.

## Stage 1: Natural Sum-Only Deterministic Bridge

Run this stage only after Stage 0B passes.

Use the passing sum-only semantic decoder checkpoint with:

```text
calculator_estimator=gumbel_concrete_interface
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
input_proj_lr=0.03
steps=300
snapshot_every=25
checkpoint_every=25
```

Primary selection metrics:

- learned calculator-result accuracy;
- answer exact;
- injection-zero and forced-random controls;
- learned-result-is-best-result fraction;
- learned-result minus best-result NLL gap.

Pair exact is report-only in this natural branch.

Stage 1 fast gate:

```text
answer exact >= 0.95
learned calculator-result accuracy >= 0.95
learned-result-is-best-result high enough to support retention
semantic decoder delta == 0.0
```

If no Stage 1 snapshot passes, stop and interpret the result as a natural
sum-only bridge negative under the passing decoder. Do not run retention from
weak checkpoints unless there is a near-gate snapshot worth testing.

## Stage 2: Relaxation-Off Retention

Run only from a Stage 1 passing or near-passing checkpoint.

Switch to:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
aux_operand_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
```

Retention success requires:

```text
final answer exact >= 0.98
final learned calculator-result accuracy >= 0.98
result-aware full-enum learned-best/result gap near 0.0
injection-zero and forced-random near chance
semantic decoder delta == 0.0
all teacher/local/expected/relaxed objectives inactive
```

## Optional Stage 3: Upstream-Open Natural Stress

Run this only if Stage 1/2 succeed cleanly.

Use the deterministic bridge with a conservative upstream LR, mirroring the
successful identifiable upstream-open stress:

```text
freeze_upstream_encoder=false
upstream_lr small and explicitly reported
semantic_decoder frozen
```

This is optional. A clean natural frozen-upstream result is more valuable than
an over-broad task that never finishes.

## Required Diagnostics

For every selected bridge or retention checkpoint, run:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

Make sure the full-enum diagnostic reports result-aware metrics for
`answer_format=sum`:

- best-result group;
- best-result group matches true sum;
- learned-result-is-best-result;
- learned-result minus best-result gap;
- true-result minus best-result gap;
- same-true-sum near-best pair count;
- effective result count.

If these metrics are missing or not summarized by the runner, add the missing
summary plumbing before interpreting the result.

## Reporting Contract

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-12-sum-only-semantic-decoder-gate.md
```

Include:

- the Stage 0A diagnosis;
- every Stage 0B candidate and why it passed or failed;
- exact commands;
- run paths;
- selected checkpoints;
- result-aware diagnostic table;
- objective weights at selected checkpoints;
- parameter movement summary for input projection, upstream, and semantic
  decoder;
- a clear go/no-go decision for natural relaxed bridge training;
- a recommendation about whether to proceed to `operand_max=99`, continue
  natural `0..19`, or return to identifiable Phase 6 closure.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push.

## Success Labels

Use one or more:

```text
sum_only_decoder_gate_positive
natural_sum_only_bridge_positive
natural_result_retention_positive
sum_only_decoder_capacity_blocker
sum_only_read_position_blocker
natural_sum_only_bridge_negative
```

## Stop Conditions

Stop early if:

- the Stage 0 gate cannot reach oracle-at-eval/result-match `>=0.98`;
- the fix requires a broad architecture search rather than a narrow wiring or
  capacity adjustment;
- bridge training would rely on oracle operands, true operand CE, hard-best CE,
  or semantic decoder movement.

The most useful negative result is a clean blocker with the next experimental
axis named precisely.
