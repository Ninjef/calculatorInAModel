# Phase 6 Seventh Task: Relaxed Bridge Replication, Stochastic Gumbel, And Upstream-Open Stress

## Mission

Turn the Phase 6 relaxed-bridge positive from a single exciting checkpoint into
a credible research claim.

The current strongest result is:

```text
In the strict semantic_decoder_only setup, deterministic hard-forward /
soft-backward Concrete training used answer loss to train an exact hard
calculator-query protocol, without true operand labels, oracle operands during
training, hard-best local-target CE, or exact expected answer-loss optimization.
The protocol then retained after the relaxed bridge was turned off.
```

That is the most end-goal-relevant Phase 6 result so far, because it is closer
to "answer loss trains the model-side tool interface" than the full-enum
hard-best local-target branch.

This task should answer three immediate questions:

```text
1. Does the deterministic relaxed-bridge result replicate across seeds?
2. Does literal stochastic Gumbel-Softmax training work, or only deterministic
   Concrete/softmax relaxation?
3. Can the same relaxed bridge tolerate carefully opened upstream parameters,
   so gradients through the calculator boundary shape more than the final
   input projection?
```

Do not broaden the task to larger operands, more digits, or new answer formats.
The fastest path to the project goal is to validate or falsify the relaxed
bridge under the current identifiable `0..19` setup before scaling.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 4 made the task identifiable with `sum_left_operand`, so the correct
  calculator action is no longer hidden behind many equivalent sums.
- Phase 4 and Phase 5 proved that answer loss can retain or complete a
  sufficiently good learned protocol after direct operand supervision is exactly
  removed.
- Phase 6 full-enum diagnostics proved the answer-NLL action landscape is sharp:
  best=true `1.000`, tie-aware true-best `1.000`, effective pairs about `1.078`,
  and true-pair probability about `0.988`.
- Phase 6 hard-best local-target training solved both full-model and strict
  `semantic_decoder_only` branches, then retained exact protocols with the
  local target exactly `0.0`.
- Phase 6 local-target decay showed a useful boundary: answer-only continuation
  did not rescue early partial protocols at steps `25` and `50`, while the
  near-gated step `75` retained exactly.
- Phase 6 exact expected answer-loss training was wired and lowered expected
  cost, but it collapsed to wrong hard actions.
- Phase 6 deterministic hard-forward / soft-backward relaxed training avoided
  that collapse and reached exact hard actions from answer loss.

Less helpful directions right now:

- Oracle-only reruns. Oracle-at-eval is a wiring gate, not progress on learned
  calculator use.
- More constant-weight hard-best local-target teaching. That result already
  passes and is now mainly a regression control.
- More simple linear local-target decay. The strict decay ladder through `150`
  steps failed to hand off cleanly.
- Broad repeats of independent-head expected answer loss. The failure mode is
  already clear: expected cost can fall while hard argmax actions become wrong.
- Scaling operands/digits before stress-testing the relaxed bridge. A larger
  task will be harder to interpret if the one-seed Phase 6 relaxed result is
  fragile.
- Calling the existing positive "Gumbel" without qualification. The successful
  branch used deterministic Concrete/softmax relaxation; actual stochastic
  Gumbel sampling remains unproven.

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
aiAgentWorkHistory/phase6/2026-05-11-gumbel-concrete-interface-bridge.md
aiAgentWorkHistory/phase6/2026-05-11-exact-expected-answer-loss-interface-discovery.md
```

Inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_gumbel_concrete_interface_bridge.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Fixed Setup

Unless a branch explicitly says otherwise, keep the strict Phase 6 identifiable
setup:

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
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Use the standard Stage 0B checkpoint as the source for semantic decoder
weights:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If absent, use the absolute path recorded in:

```text
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
```

Use a new run root:

```text
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream
```

## Critical Guardrail

This task is about relaxed answer-loss training, not local-target teaching.

Allowed for training:

- answer loss through `calculator_estimator=gumbel_concrete_interface`;
- deterministic Concrete/softmax relaxation;
- straight-through stochastic Gumbel-Softmax samples;
- hard-forward / soft-backward calculator signals;
- temperature schedules;
- entropy bonuses if clearly labeled and reported;
- upstream parameter movement only in the upstream-open branch.

Forbidden for training:

- true operand CE;
- true sum targets;
- hard-best pair CE;
- soft target CE distilled from full-enum answer losses;
- `calculator_estimator=identifiable_full_enum_local_target`;
- `calculator_estimator=full_enum_expected_answer_loss`;
- oracle operands during training;
- semantic decoder movement.

True operands, oracle-at-eval, hard-best full-enum pairs, and direct aux CE may
be used only for diagnostics, parity checks, and interpretation.

## Stage 0: Regression And Gradient Gates

Do not repeat every old oracle diagnostic. Run only the gates needed to make
the new branches interpretable.

### Gate A: Deterministic Gradient Gate Regression

Rerun the existing one-step gate for deterministic mode:

```text
scripts/run_phase6_gumbel_concrete_interface_bridge.py stage0-gradient-gate
```

Use:

```text
samples=128
temperature=2.0
mode=deterministic
hard_forward=true
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
```

Required report:

- oracle-at-eval exact;
- injection-zero exact;
- forced-random exact;
- initial answer loss;
- initial hard operand/pair/calc;
- entropy/effective pairs;
- best-pair probability before and after one step;
- gradient cosine versus diagnostic hard-best CE;
- input-proj/upstream/semantic parameter deltas;
- semantic decoder grad exactly `0.0`.

Pass criterion:

```text
best-pair probability delta > 0.0
gradient cosine > 0.0
input-proj delta > 0.0
upstream delta == 0.0
semantic decoder grad/delta == 0.0
```

### Gate B: Stochastic Gumbel Gradient Gate

Run the same one-step gate with:

```text
mode=gumbel
temperature=2.0
```

Because this is stochastic, run at least three gate seeds and summarize mean,
min, and max for:

- best-pair probability delta;
- gradient cosine;
- input-proj delta;
- hard pair/calc at initialization.

Pass criterion:

```text
mean best-pair probability delta > 0.0
at least 2/3 gate seeds have positive gradient cosine
semantic decoder grad/delta == 0.0 for every gate seed
```

If Gate B fails while Gate A passes, keep the task alive but label later
success claims as deterministic Concrete only.

## Stage 1: Deterministic Concrete Replication

Replicate the successful deterministic branch across effective seeds
`2`, `4`, and `5`. In the current `overfit_one_batch.py` convention, these are
typically CLI seeds `0`, `2`, and `3`; record the effective seed from each run
name and metrics file rather than relying on memory.

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

- Find the first checkpoint in each seed with fast-gate
  normal/operand/pair/calc at least `0.95`.
- Also identify the best protocol checkpoint by fast-gate pair/calc.
- If a seed never reaches `0.95`, run full diagnostics on its best checkpoint
  and label it as a replication failure.

Pass criterion for deterministic replication:

```text
At least 2/3 seeds reach fast-gate operand/pair/calc >= 0.95 during Stage 1,
and every passing seed has a selected checkpoint with canonical and private
operand/pair/calc >= 0.98 and full-enum learned-best >= 0.98.
```

If only seed `2` passes, stop before upstream-open claims and treat the result
as fragile. The next task should then focus on stabilization, not scaling.

## Stage 2: Relaxation-Off Retention For Replicated Seeds

For each Stage 1 passing seed, continue from:

- the first qualifying Stage 1 checkpoint; and
- the best Stage 1 checkpoint if different.

Use:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
```

Required diagnostics on selected retained checkpoints:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

Retention pass criterion:

```text
canonical operand/pair/calc >= 0.99
private answer/operand/pair/calc >= 0.99
full-enum learned-minus-true gap == 0.0 or near numerical zero
full-enum learned-minus-best gap == 0.0 or near numerical zero
learned-best fraction >= 0.99
aux/direct/local/expected/relaxed training objectives inactive in retained run
semantic decoder delta == 0.0
```

Dense checkpointing remains mandatory. If final checkpoints drift, select and
diagnose the best retained checkpoint, but report final drift honestly.

## Stage 3: Literal Stochastic Gumbel Training

Only run this stage if either:

- Stage 0 Gate B passes; or
- the goal is to document a clear stochastic-Gumbel negative against a
  replicated deterministic positive.

Start with effective seed `2`. If it reaches the `0.95` fast-gate threshold,
replicate on the other deterministic-passing seeds.

Primary stochastic branch:

```text
calculator_estimator=gumbel_concrete_interface
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
input_proj_lr=0.03
steps=300
snapshot_every=25
checkpoint_every=25
relaxed_calculator_mode=gumbel
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
relaxed_calculator_entropy_weight=0.0
```

If the primary stochastic branch collapses early while deterministic replicates,
run one stabilization branch only:

```text
relaxed_calculator_temperature=1.5
relaxed_calculator_final_temperature=0.5
relaxed_calculator_entropy_weight=0.01
relaxed_calculator_entropy_decay_steps=200
```

Do not turn this into a broad stochastic sweep.

Interpretation:

- If stochastic branches pass and retain, the project has an actual
  Gumbel-Softmax positive.
- If deterministic passes but stochastic fails, the project has a Concrete
  relaxation positive and a stochastic-sampling stability negative.
- If both fail to replicate, the sixth task result was likely a fragile
  optimization event and the next work should diagnose temperature/LR/checkpoint
  sensitivity before any scale-up.

## Stage 4: Upstream-Open Relaxed Bridge Stress

Run this only after deterministic Stage 1/2 replication passes on at least
two seeds.

Purpose:

```text
Test whether the relaxed bridge can train with upstream parameters open, rather
than only fitting calculator_hook.input_proj on frozen random features.
```

Primary upstream-open branch:

```text
calculator_estimator=gumbel_concrete_interface
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=false
trainable=calculator_hook.input_proj plus upstream
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
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

Start with effective seed `2`. If it reaches fast-gate operand/pair/calc
`>=0.95`, run relaxation-off retention from the first qualifying and best
checkpoints. For retention, run both:

```text
freeze_upstream_encoder=true
freeze_upstream_encoder=false with upstream_lr=0.00003
```

Required upstream-open reporting:

- input-proj parameter delta from Stage 1 step `0`;
- upstream parameter delta from Stage 1 step `0`;
- number of upstream tensors changed;
- semantic decoder delta exactly `0.0`;
- whether upstream movement improved, preserved, or destabilized the protocol
  relative to frozen-upstream deterministic seed `2`;
- dense snapshot drift, not only final performance.

Pass criterion for an upstream-open positive:

```text
Stage 1 selected checkpoint reaches operand/pair/calc >= 0.95.
Relaxation-off retention reaches canonical/private operand/pair/calc >= 0.99.
Full-enum learned-minus-true/best gaps are 0.0 or near numerical zero.
Upstream parameter delta > 0.0.
Semantic decoder delta == 0.0.
```

If upstream-open fails while frozen-upstream deterministic replication passes,
interpret this as a stability/credit-assignment limitation, not as a failure of
the relaxed bridge itself.

## Required Summary Tables

The final report should include:

1. Stage 0 gradient-gate table:

```text
mode, seed, best-pair probability delta, gradient cosine, input-proj delta,
upstream delta, semantic grad/delta
```

2. Stage 1 deterministic replication table:

```text
effective seed, first gate step, best step, best fast normal/operand/pair/calc,
final fast normal/operand/pair/calc, final eval, selected checkpoint
```

3. Stage 2 retention table:

```text
effective seed, source checkpoint, final/best retained checkpoint, canonical
operand/pair/calc, private answer/operand/pair/calc, full-enum gaps,
learned-best, final objective weights
```

4. Stochastic Gumbel table:

```text
branch, effective seed, gate status, best fast protocol metrics, retention
status if run, interpretation label
```

5. Upstream-open table:

```text
effective seed, trainable groups, upstream delta, input-proj delta, best protocol
metrics, retention metrics, drift notes
```

## Success Definitions

### Deterministic Concrete Replication Positive

At least two effective seeds learn and retain an exact or near-exact hard
calculator-query protocol through the deterministic relaxed bridge, with all
teacher/local/expected objectives absent and semantic decoder movement `0.0`.

### Stochastic Gumbel Positive

At least one stochastic Gumbel branch learns and retains the hard protocol under
the same guardrails. Replication across deterministic-passing seeds is stronger
but not required for the first stochastic positive.

### Upstream-Open Positive

The deterministic relaxed bridge succeeds with upstream parameters open,
upstream weights move measurably, semantic decoder weights do not move, and the
hard protocol survives relaxation-off retention.

### Negative Worth Keeping

A negative is still valuable if it cleanly separates mechanisms:

- deterministic Concrete replicates but stochastic Gumbel fails;
- frozen-upstream deterministic replicates but upstream-open destabilizes;
- Stage 1 reaches exact protocol but relaxation-off retention drifts;
- a seed fails despite passing oracle and gradient gates.

## Reporting Contract

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-11-relaxed-bridge-replication-stochastic-upstream.md
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
  `deterministic_concrete_positive`,
  `stochastic_gumbel_positive`,
  `stochastic_gumbel_negative`,
  `upstream_open_positive`,
  `upstream_open_instability`, or
  `relaxed_bridge_replication_failure`.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push following `CLAUDE.md`.
