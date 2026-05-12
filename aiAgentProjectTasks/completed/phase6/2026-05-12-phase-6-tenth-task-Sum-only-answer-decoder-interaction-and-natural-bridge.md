# Phase 6 Tenth Task: Sum-Only Answer Decoder Interaction And Natural Bridge

## Mission

Fix the natural sum-only decoder/readout blocker with the smallest targeted
model change, then rerun the natural deterministic relaxed bridge only if the
new decoder gate passes.

The current Phase 6 conclusion is strong but still artificial:

```text
In the identifiable sum_left_operand setup, deterministic hard-forward /
soft-backward Concrete answer-loss training can discover hard calculator
actions without true operand labels, hard-best CE, oracle operands, or semantic
decoder movement, and those actions retain after the relaxation is off.
```

The latest natural sum-only attempts did not test that bridge fairly. They
stopped at Stage 0 because the strict sum-only semantic decoder could not decode
oracle calculator results reliably enough:

```text
oracle-at-eval exact ~= 0.93
forced true result exact ~= 0.93
full-enum best result group matches true sum ~= 0.91
```

This task should answer:

```text
Can an interaction-capable sum-only answer decoder clear the natural Stage 0
gate, and, if so, does deterministic Concrete answer-loss training learn a
correct calculator result protocol in the natural underidentified sum-only
task?
```

## Why This Is The Next Best Task

Helpful findings so far:

- Phase 4 proved the architecture can support a true learned calculator-query
  protocol when the answer target makes operand identity identifiable.
- Phase 5 showed answer-only training can preserve or complete a partially
  taught protocol, but plain no-handoff answer-only training does not discover
  the protocol.
- Phase 6 hard-best local targets proved the frozen answer decoder creates a
  sharp answer-derived interface target in the identifiable task.
- Phase 6 strict random-upstream local-target training solved the interface
  with only the frozen semantic decoder loaded.
- Phase 6 deterministic Concrete is the closest result to the end goal: answer
  loss itself trained the calculator interface, with no true operand CE,
  hard-best CE, full-enum expected-loss objective, or oracle operands during
  training.
- Deterministic Concrete replicated across effective seeds `2`, `4`, and `5`,
  retained after the relaxation was off, and tolerated modest upstream movement.

Less helpful directions right now:

- More oracle-only success in already healthy `sum_left_operand` wiring.
- More hard-best local-target teaching in the identifiable task. It works and
  is now mainly a control.
- More simple linear local-target decay. The tested decay ladder did not hand
  off cleanly.
- More independent-head exact expected answer-loss sweeps. Expected loss fell,
  but hard argmax actions collapsed to wrong protocols.
- More literal stochastic Gumbel before the NaN/variance problem is fixed.
- More tiny-model sum-only capacity ladders. The previous ladder did not clear
  the gate, and the code suggests a more structural readout limitation.
- Scaling to `operand_max=99` before natural `0..19` has a healthy decoder gate.

The fastest path to the project goal is therefore:

```text
repair the natural sum-only decoder gate -> rerun the deterministic Concrete
bridge in 0..19 -> only then decide whether to scale or close Phase 6.
```

## Key Hypothesis

The current strict answer decoder is likely underexpressive for natural
sum-only output.

In `src/model.py`, `_answer_bottleneck_logits` currently does:

```text
sum_left_operand: decoder_h = selected_signal + offset_h + selected_signal * offset_h
sum:              decoder_h = selected_signal + offset_h
```

For multi-token sum answers, an additive-only linear decoder has to emit
position-specific digits from a result class and an answer-position embedding.
That factorization can be too weak because the effect of answer position is not
allowed to depend directly on the calculator result class. The successful
`sum_left_operand` path already uses the simplest interaction term:

```text
selected_signal * offset_h
```

This task should test whether giving sum-only the same opt-in interaction fixes
the natural decoder gate.

Do not present the decoder gate itself as learned calculator use. It is a
readout health prerequisite.

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
aiAgentWorkHistory/phase6/2026-05-12-sum-only-semantic-decoder-gate.md
```

Inspect:

```text
src/model.py
src/data.py
scripts/overfit_one_batch.py
scripts/run_phase6_sum_only_semantic_decoder_gate.py
scripts/run_phase6_natural_sum_only_relaxed_bridge.py
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Critical Guardrail

Oracle operands, forced true result classes, and oracle-at-eval checks are
wiring gates only.

Natural sum-only success must be reported with result-level learned-interface
metrics, not pair-identifiability metrics. In the natural sum-only task, many
operand pairs share the same sum, so pair exact is not the main success target.

Required learned-interface metrics after bridge training:

- learned calculator-result accuracy;
- normal answer exact;
- full-enum learned-result best fraction;
- learned-result minus best-result NLL gap;
- best-result group matches true sum;
- injection-zero, forced-zero, and forced-random controls;
- semantic decoder movement exactly `0.0`;
- final objective weights showing no true operand CE, no hard-best CE, no
  full-enum local target, no expected-answer-loss objective, and no oracle
  operands during bridge training.

## Stage 0: Implement An Opt-In Sum-Only Interaction Readout

Add a backward-compatible option for the strict answer decoder to use an
interaction term in sum-only mode.

Suggested shape:

```text
--answer-decoder-interaction none | product
```

Default must preserve existing behavior:

```text
answer_decoder_interaction=none
```

Behavior:

```text
none:
  decoder_h = selected_signal + offset_h

product:
  decoder_h = selected_signal + offset_h + selected_signal * offset_h
```

For compatibility, existing `sum_left_operand` checkpoints may keep their
current implicit product behavior, or the implementation may migrate the logic
so `sum_left_operand` defaults to product while `sum` defaults to none. Be
explicit in configs and summaries either way.

Requirements:

- Serialize the new option in checkpoint configs and summary metrics.
- Make semantic-decoder load scope include any new parameters only if new
  parameters are added. Prefer no new parameters if the product interaction is
  enough.
- Add or update tests for:
  - default sum-only behavior stays additive;
  - product interaction changes answer-decoder hidden states when requested;
  - old checkpoints without the new config field load with the old default.

## Stage 1: Sum-Only Decoder Gate

Train a fresh oracle sum-only semantic decoder with the interaction enabled.

Fixed natural setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
answer_format=sum
calculator_output_format=sum
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
calculator_action_head=independent_operands
answer_decoder_interaction=product
oracle_train=true
```

Start narrow:

```text
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
batch_size=400
steps=1000 to 2000
snapshot_every=250
checkpoint_every=250
eval_samples=400
```

Select checkpoints by all-400 gate metrics, not by final loss alone.

Required gate:

```text
oracle-at-eval exact >= 0.98
forced true result exact >= 0.98
full-enum best-result group matches true sum >= 0.98
injection-zero and forced-random near chance
```

If the tiny product decoder does not pass, run only one narrow fallback:

```text
n_embd=32
n_head=2
n_layer=2
answer_decoder_interaction=product
```

Stop if that also fails. Do not run a broad oracle capacity sweep.

Deliverables:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.json
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.md
```

## Stage 2: Natural Deterministic Concrete Bridge

Run this stage only if Stage 1 passes.

Use the passing semantic decoder checkpoint with:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
calculator_estimator=gumbel_concrete_interface
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
```

Run effective seed `2` first. If it passes result-level gates, replicate on
effective seeds `4` and `5`. Keep dense snapshots and select by learned
calculator-result accuracy plus normal answer exact.

Bridge pass threshold:

```text
normal answer exact >= 0.95
learned calculator-result accuracy >= 0.95
full-enum learned-result best fraction >= 0.95
mean learned-result minus best-result gap near 0.0
semantic decoder delta == 0.0
```

Pair exact may be low and should not fail the natural branch when result-level
metrics pass.

## Stage 3: Relaxation-Off Retention

Run this stage only for Stage 2 checkpoints that pass or near-pass.

Continue from the selected Stage 2 checkpoint with:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
oracle_train=false
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed objective inactive
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
input_proj_lr=0.0003
steps=1000
```

Retention pass threshold:

```text
normal answer exact >= 0.98
learned calculator-result accuracy >= 0.98
full-enum learned-result best fraction >= 0.98
semantic decoder delta == 0.0
```

If Stage 2 is near-gated but not exact, it is acceptable to run one
relaxation-off continuation to see whether answer-only retention completes the
result protocol, matching the identifiable Phase 6 pattern.

## Stage 4: Reporting And Decision

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/<date>-sum-only-interaction-decoder-natural-bridge.md
```

Move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Commit and push after the experiment is complete.

Decision labels:

```text
sum_only_interaction_gate_positive
sum_only_interaction_gate_negative
natural_deterministic_concrete_result_positive
natural_deterministic_concrete_result_negative
natural_retention_positive
natural_retention_negative
```

Recommended interpretation:

- If Stage 1 fails, Phase 6 should close with a decoder/readout blocker and the
  next phase should focus on semantic decoder architecture, not bridge training.
- If Stage 1 passes but Stage 2 fails, Phase 6 has a real natural bridge
  negative under deterministic Concrete.
- If Stage 2 and Stage 3 pass, Phase 6 has reached the first natural sum-only
  answer-loss learned-interface positive at `0..19`, and the next phase can
  scale carefully to `operand_max=99` or test upstream-open natural retention.

## Validation

At minimum:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_phase6_sum_only_semantic_decoder_gate.py scripts/run_phase6_natural_sum_only_relaxed_bridge.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Add any new runner to the compile command if this task creates one.
