# Phase 6 Eighth Task: Natural Sum-Only Relaxed Bridge

## Mission

Test whether the strongest current Phase 6 method still works when the answer
target is the natural addition answer:

```text
AA+BB=SSS<eos>
```

instead of the deliberately identifiable Phase 4/5 target:

```text
AA+BB=SSSAA<eos>
```

The specific question is:

```text
Can deterministic hard-forward / soft-backward Concrete answer-loss training
learn calculator inputs whose calculator result is correct, with no true
operand labels, no hard-best local target, no expected-answer-loss objective,
no oracle operands during training, and the semantic decoder frozen?
```

For this task, exact true-operand recovery is not the primary success metric.
In the natural sum-only task, many action pairs can be equally valid because
the calculator output is only the sum. The required learned-interface behavior
is therefore:

```text
learned calculator-result accuracy, answer exact, calculator dependence, and
retention after the relaxed bridge is off.
```

## Why This Is The Next Best Task

Helpful evidence so far:

- Phase 4 made the protocol identifiable with `sum_left_operand` and showed
  that a true calculator-query protocol can be taught and retained.
- Phase 5 showed that answer-only training can preserve or complete a near-good
  protocol, but broad no-handoff answer-only discovery did not work.
- Phase 6 hard-best local-target training proved that frozen answer loss
  contains a sharp calculator-action signal in the identifiable setup.
- Phase 6 exact expected answer loss was wired correctly but collapsed to wrong
  hard argmax actions, so reducing expected cost alone is not enough.
- Phase 6 deterministic hard-forward / soft-backward Concrete training is now
  the strongest result: it replicated across effective seeds `2`, `4`, and `5`,
  retained or completed exact hard protocols after the relaxation was turned
  off, and tolerated modest upstream movement while the semantic decoder stayed
  fixed.

Less helpful directions right now:

- More oracle-only runs. They remain wiring checks, not research progress.
- More hard-best local-target teaching. That branch already succeeded and is
  now a control, not the frontier.
- More simple linear local-target decay. The tested decay ladder failed through
  `150` steps.
- More independent-head exact expected-answer-loss sweeps. The failure mode is
  clear: expected cost can fall while hard actions are wrong.
- More literal stochastic Gumbel sampling before fixing its numerical
  instability. The stochastic gate was positive, but training went to `NaN` and
  stayed near chance.
- Scaling to larger operands before testing whether the relaxed bridge survives
  the natural answer target. A `0..99` run is valuable, but it is easier to
  interpret after this smaller end-goal-like check.

This task deliberately changes only one conceptual thing: remove the extra
left-operand answer suffix and require useful calculator-result discovery in
the original sum-only setting.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-10-phase-6-overarching_plan-Identifiable-local-interface-discovery.md
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-12-relaxed-bridge-replication-stochastic-upstream.md
```

Inspect:

```text
src/data.py
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Fixed Setup

Unless a branch explicitly says otherwise:

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

Use the existing strict sum-only oracle semantic decoder checkpoint as the
first wiring candidate:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

If that checkpoint gives weak oracle-at-eval under the current diagnostics,
train a fresh oracle sum-only semantic decoder as a Stage 0 wiring artifact.
Do not count that oracle training as progress.

Use a new run root:

```text
runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge
```

## Critical Guardrail

This task is about relaxed answer-loss training in the natural sum-only task.

Allowed for training:

- `calculator_estimator=gumbel_concrete_interface`;
- deterministic Concrete/softmax relaxation;
- hard-forward / soft-backward calculator signals;
- temperature schedules;
- answer loss only;
- optional upstream movement in the upstream-open branch.

Forbidden for training:

- true operand CE;
- true sum CE outside the normal answer target;
- hard-best pair CE;
- soft targets distilled from full-enum answer losses;
- `calculator_estimator=identifiable_full_enum_local_target`;
- `calculator_estimator=full_enum_expected_answer_loss`;
- oracle operands during training;
- semantic decoder movement.

True operands, true sums, full-enum best actions, and oracle-at-eval may be
used only for diagnostics and interpretation.

## Implementation Requirements

Add a dedicated runner, for example:

```text
scripts/run_phase6_natural_sum_only_relaxed_bridge.py
```

It can reuse structure from:

```text
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
```

but it must parameterize or replace all hardcoded `sum_left_operand` settings
with:

```text
answer_format=sum
calculator_output_format=sum
```

Add or adapt summary helpers so sum-only diagnostics include result-level
action metrics. Existing exact-pair metrics are still useful for reporting, but
they are not enough for this task because same-sum actions are equivalent under
the natural answer target.

At minimum, every selected checkpoint summary should report:

- hard learned calculator-result accuracy;
- hard learned pair exact, for reporting only;
- full-enum best-result-match fraction;
- learned-result-is-best-result fraction;
- learned-minus-best-result NLL gap when a result-aware grouping is available;
- number of same-sum near-best actions in Stage 0 landscape summaries;
- final objective weights, including relaxed entropy weight;
- semantic decoder parameter delta.

If `scripts/run_full_enum_action_loss_diagnostic.py` only reports pair-level
`learned-best` for the current checkpoint, extend the runner's postprocessing
or add a small companion diagnostic rather than interpreting pair-level
`learned-best` as failure in the sum-only setting.

## Stage 0: Sum-Only Wiring And Landscape Gate

Before training, verify that the sum-only decoder is healthy and that the
natural task is interpretable.

Run a compact gate under:

```text
answer_format=sum
calculator_output_format=sum
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
```

Required report on a fixed 128-sample batch:

- built-in eval exact;
- oracle-at-eval exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- initial hard learned answer exact;
- initial hard learned calculator-result accuracy;
- full-enum best calculator-result matches true sum;
- full-enum true-pair best fraction, for reporting only;
- number of near-best same-sum action pairs;
- effective action pairs under `softmax(-NLL/T)` for a reasonable diagnostic
  temperature;
- semantic decoder grad/delta exactly `0.0`.

Pass criterion:

```text
oracle-at-eval exact >= 0.98, or train/validate a fresh oracle sum-only
semantic decoder before proceeding.
injection-zero and forced-random remain near chance.
full-enum best calculator-result matches true sum on nearly all samples.
semantic decoder delta == 0.0.
```

Do not require the full-enum best action pair to equal the true operands. In
the sum-only setting, that is intentionally underidentified.

## Stage 1: Frozen-Upstream Deterministic Concrete Sum-Only Training

Start with effective seed `2`.

Use:

```text
calculator_estimator=gumbel_concrete_interface
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
steps=300
snapshot_every=25
checkpoint_every=25
snapshot_samples=128
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
relaxed_calculator_entropy_weight=0.0
```

Selection:

- first checkpoint with fast-gate normal exact and learned calculator-result
  accuracy both `>=0.95`;
- best checkpoint by learned calculator-result accuracy;
- final checkpoint, even if it drifts.

Required selected-checkpoint diagnostics:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

Interpretation for Stage 1:

- Success is learned calculator-result accuracy, not true-pair exact.
- Learned pair exact should be reported but should not be used as the gate.
- If answer exact is high but learned calculator-result accuracy is low, treat
  that as a bypass/private-code failure unless counterfactuals prove otherwise.

## Stage 2: Relaxation-Off Sum-Only Retention

If Stage 1 reaches the `>=0.95` fast gate, continue from the first qualifying
checkpoint and the best checkpoint if different.

Use:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
```

Retention pass criterion:

```text
normal answer exact >= 0.99
learned calculator-result accuracy >= 0.99
injection-zero near chance
forced-random near chance
oracle-at-eval remains healthy
semantic decoder delta == 0.0
all teacher/local/expected/relaxed objectives inactive
```

If final checkpoints drift, diagnose the best retained checkpoint and report
the drift honestly.

## Stage 3: Seed Replication

If effective seed `2` passes Stage 1 and Stage 2, replicate frozen-upstream
deterministic Concrete training and retention on effective seeds `4` and `5`
using the same CLI seed convention as the seventh task.

Pass criterion:

```text
At least 2/3 effective seeds retain or complete to answer exact >= 0.99 and
learned calculator-result accuracy >= 0.99 after the relaxation is off.
```

If only seed `2` passes, stop and label the natural sum-only bridge as fragile.
The next task should stabilize the natural target before scaling.

## Stage 4: Upstream-Open Natural Sum-Only Stress

Run this only after Stage 3 passes on at least two seeds.

Start with effective seed `2`:

```text
calculator_estimator=gumbel_concrete_interface
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=false
trainable=calculator_hook.input_proj plus upstream
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
upstream_lr=0.00003
steps=300
snapshot_every=25
checkpoint_every=25
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
```

If it reaches the Stage 1 gate, run relaxation-off retention from selected
checkpoints with:

- upstream frozen; and
- upstream still open at `upstream_lr=0.00003`.

Report:

- input-proj parameter delta;
- upstream parameter delta;
- upstream tensors changed;
- semantic decoder delta exactly `0.0`;
- whether upstream movement improves, preserves, or destabilizes learned
  calculator-result accuracy.

## Do Not Scale Yet Unless This Passes

If the natural `0..19` sum-only bridge passes seed replication and retention,
the next task should scale the deterministic bridge to full two-digit
`operand_max=99`.

If it fails, do not jump to `0..99`. First diagnose whether the failure is:

- weak sum-only semantic decoder wiring;
- underidentified action landscape causing unstable hard actions;
- temperature/LR sensitivity;
- answer-only retention drift;
- upstream-open instability.

## Required Summary Tables

The completion report should include:

1. Stage 0 wiring/landscape gate:

```text
checkpoint, oracle, injection-zero, forced-random, best-result-match,
true-pair-best, same-sum near-best count, semantic delta
```

2. Stage 1 natural sum-only training:

```text
effective seed, first gate step, best step, best normal exact,
best learned calc accuracy, learned pair exact, final metrics, selected checkpoint
```

3. Stage 2 retention:

```text
effective seed, source checkpoint, retained checkpoint, normal exact,
learned calc accuracy, injection-zero, forced-random, oracle, objective weights
```

4. Full diagnostics:

```text
canonical normal/oracle/injection-zero/forced-random, learned operand/pair/calc,
private answer/operand/pair/calc, full-enum learned-result/best-result summaries
```

5. Upstream-open stress, if run:

```text
effective seed, trainable groups, input-proj delta, upstream delta,
best learned calc accuracy, retention learned calc accuracy, drift notes
```

## Success Definitions

### Natural Sum-Only Positive

Deterministic Concrete answer-loss training learns hard calculator actions whose
calculator result matches the true sum, and relaxation-off answer-only
retention preserves or completes that behavior with all auxiliary, local,
expected, relaxed, and oracle training signals inactive.

### Natural Sum-Only Seed-Robust Positive

At least two effective seeds pass the retained learned-calculator-result gate,
with semantic decoder movement exactly `0.0` and calculator counterfactuals
showing dependence on the learned calculator path.

### Upstream-Open Natural Positive

The natural sum-only bridge passes with upstream parameters open, upstream
weights move measurably, semantic decoder weights do not move, and the learned
calculator-result protocol survives relaxation-off retention.

### Useful Negative

A negative is still worth keeping if it cleanly separates mechanisms:

- deterministic Concrete works only in the identifiable target;
- sum-only training gets high answer exact but low calculator-result accuracy;
- sum-only Stage 1 learns a useful result protocol but retention drifts;
- frozen-upstream passes but upstream-open destabilizes;
- the old sum-only semantic decoder is too weak and needs a refreshed wiring
  checkpoint before interpretation.

## Reporting Contract

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-12-natural-sum-only-relaxed-bridge.md
```

Record:

- exact commands;
- run paths;
- selected checkpoints;
- all seeds and effective seeds;
- all final objective weights;
- dense snapshot selection rule;
- canonical/private/full-enum diagnostics;
- parameter movement summaries;
- clear interpretation labels:
  `natural_sum_only_positive`,
  `natural_sum_only_fragile`,
  `natural_sum_only_negative`,
  `natural_sum_only_retention_failure`,
  `natural_sum_only_upstream_open_positive`, or
  `natural_sum_only_upstream_open_instability`.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push following `CLAUDE.md`.
