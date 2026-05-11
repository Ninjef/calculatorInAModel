# Phase 6 Sixth Task: Gumbel/Concrete Hard-Forward Interface Bridge

## Mission

Test whether a differentiable relaxation at the calculator interface can let
answer loss train the strict identifiable calculator-query protocol without
direct true-operand labels, oracle operands, or answer-derived hard-best
pseudo-label CE.

Phase 6 has now separated three facts:

```text
1. The identifiable answer landscape is sharp: full-enum best action equals the
   true operand pair.
2. A hard-best local target distilled from that landscape can teach and retain
   the true protocol, even with semantic_decoder_only load scope.
3. Direct independent-head expected answer-loss optimization lowers expected
   cost but collapses to wrong hard actions.
```

The next best task is therefore a bridge between pure expected-cost training
and hard-best local-target teaching:

```text
Use a hard-forward / soft-backward Concrete calculator signal, so the forward
path still uses a hard calculator action while gradients flow through a soft
distribution over calculator inputs and outputs.
```

This is a biased relaxation, but it removes the hard-best CE target and asks
whether answer loss itself can shape the interface when given a smoother local
gradient.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 4 made the task identifiable with `sum_left_operand`, so the desired
  action is not hidden behind many equivalent sums.
- Phase 5 showed plain answer-only no-handoff training is not enough, but
  answer-only continuation can preserve or complete a sufficiently good partial
  protocol.
- Phase 6 Stage 0 proved the full-enum answer-NLL argmin is sharp:
  best=true `1.000`, effective pairs about `1.078`, true-pair probability about
  `0.988`.
- Phase 6 hard-best local-target teaching solved both full-model and strict
  `semantic_decoder_only` branches, then retained the protocol after local
  target weight was exactly `0.0`.
- Phase 6 minimum-handoff work showed answer-only continuation does not rescue
  early partial protocols at steps `25` and `50`, but does retain from the near
  gated step `75`.
- Phase 6 expected answer-loss work proved the exact expected-cost objective is
  wired and has large gradients, but it collapses probability mass to wrong
  hard actions.

Less helpful directions right now:

- Oracle-only reruns. Oracle-at-eval is a wiring gate, not a learned-interface
  result.
- More hard-best local-target teaching at constant weight. That branch already
  passes and is now mostly a regression check.
- More simple linear local-target decay. The strict decay ladder through `150`
  steps failed to hand off cleanly.
- Broad repeats of independent-head expected answer loss. The failure mode was
  clear: expected loss decreased, entropy collapsed, and hard argmax actions
  stayed wrong.
- Upstream-open expected-loss runs as the immediate next move. Hard-best
  strict frozen-upstream teaching already showed the frozen random readout is
  sufficient when the interface receives a usable target, so the sharper
  question is the gradient/objective, not representation capacity.
- Larger operand ranges, more digits, or multi-task settings. The `0..19`
  identifiable setup is still the fastest honest test bed.

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
aiAgentWorkHistory/phase6/2026-05-11-exact-expected-answer-loss-interface-discovery.md
```

Inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_strict_random_upstream_local_target.py
scripts/run_phase6_strict_local_target_decay_boundary.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_private_protocol.py
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
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge
```

## Critical Guardrail

This task is not allowed to train on true operands or hard-best local targets.

Allowed for training:

- answer loss through a relaxed calculator signal;
- soft operand distributions derived from current model logits;
- deterministic Concrete/softmax relaxation;
- optional Gumbel-Softmax straight-through samples;
- temperature schedules;
- entropy bonuses or floors if clearly labeled.

Forbidden for training:

- true operand CE;
- true sum targets;
- hard-best pair CE;
- soft target CE distilled from `softmax(-answer_nll / T)`;
- expected answer-loss objective from the fifth task, except as a diagnostic
  comparison;
- oracle operands during training;
- semantic decoder movement.

True operands and hard-best full-enum pairs may be used only in diagnostics,
gradient-alignment gates, and reporting.

## Implementation Requirements

Add a narrowly named estimator, for example:

```text
calculator_estimator=gumbel_concrete_interface
```

or:

```text
calculator_estimator=relaxed_calculator_interface
```

The implementation should support the independent operand heads first. Joint
pair support is not needed for this task.

Required behavior:

- Read `a_logits` and `b_logits` from the same `operand_spans` interface used
  by the successful strict hard-best branch.
- Construct soft operand distributions:

```text
p_a = softmax(a_logits / temperature)
p_b = softmax(b_logits / temperature)
```

- Construct a differentiable soft result distribution over sums by convolving
  the independent operand distributions:

```text
p_sum[s] = sum_{a+b=s} p_a[a] * p_b[b]
```

- For `calculator_output_format=sum_left_operand`, construct the soft
  calculator signal as:

```text
concat(p_sum, p_a)
```

- Keep the primary forward path hard. Use either deterministic argmax or
  straight-through Gumbel-Softmax one-hot operands to produce the hard
  calculator action and hard calculator signal.
- Combine hard forward and soft backward with the standard straight-through
  pattern:

```text
calculator_signal = hard_signal.detach() + soft_signal - soft_signal.detach()
```

- Eval/diagnostics must still use hard learned actions and report learned
  operand/pair/calc metrics.
- The semantic decoder and `calculator_hook.output_proj` must remain frozen
  when `freeze_semantic_decoder=true`.

Recommended CLI knobs:

```text
--relaxed-calculator-temperature
--relaxed-calculator-final-temperature
--relaxed-calculator-temperature-decay-steps
--relaxed-calculator-mode deterministic|gumbel
--relaxed-calculator-hard-forward
--relaxed-calculator-entropy-weight
--relaxed-calculator-entropy-decay-steps
```

Use deterministic mode first. Gumbel sampling is useful only after the
deterministic straight-through gradient gate is understood.

Required metrics:

- relaxed temperature and final temperature;
- relaxed mode;
- operand entropy and effective pair count;
- hard learned operand/pair/calc accuracy;
- answer loss under the hard forward path;
- optional soft-forward answer loss as a diagnostic only;
- full-enum learned/true/best NLLs on selected checkpoints;
- learned-minus-true and learned-minus-best gaps;
- hard learned-best fraction;
- semantic decoder delta;
- input-proj and upstream parameter deltas.

## Stage 0: Gradient Alignment Gate

Before a full training ladder, prove that the relaxed answer-loss gradient
points in a useful direction under the strict `semantic_decoder_only`
initialization.

Run on a fixed 128-sample batch with:

```text
freeze_upstream_encoder=true
answer_loss_weight=1.0
relaxed calculator objective active
aux/local/expected/anchor weights all 0.0
```

Report:

- oracle-at-eval exact `1.000` as a wiring gate;
- injection-zero and forced-random near chance;
- initial hard learned operand/pair/calc;
- initial answer loss;
- initial operand entropy and effective pairs;
- full-enum best=true fraction, for reporting only;
- one-step input-proj delta;
- upstream delta `0.0`;
- semantic decoder grad/delta `0.0`;
- change in probability assigned to the full-enum best pair after one step;
- change in hard learned-best fraction after one step, if any;
- cosine similarity between the relaxed answer-loss gradient and the hard-best
  local-target CE gradient, for diagnostics only.

Pass criteria to continue:

```text
semantic_decoder_delta == 0.0
input_proj_delta > 0.0
best_pair_probability increases on the fixed batch
gradient cosine is positive or the one-step hard/soft diagnostics improve
```

If the relaxed gradient does not move probability toward the full-enum best
pair even on the fixed batch, stop and diagnose the relaxation before launching
long training.

## Stage 1: Strict Frozen-Upstream Relaxed Training

Primary question:

```text
Can answer loss through a hard-forward / soft-backward calculator relaxation
train the strict random/new interface to a true hard calculator-query protocol?
```

Use:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
steps=300
snapshot_every=25
checkpoint_every=25
snapshot_samples=128
input_proj_lr=0.03
```

Run a compact ladder, stopping early if a branch reaches a fast-gate
operand/pair/calc threshold at or above `0.90`.

| Branch | Mode | Temperature | Entropy | Reason |
| --- | --- | --- | --- | --- |
| A | deterministic | `2.0 -> 0.5` over `300` | `0.0` | Smoothest low-variance bridge |
| B | deterministic | `1.0 -> 0.25` over `300` | `0.0` | Sharper bridge if A is too soft |
| C | deterministic | `1.0 -> 0.25` over `300` | small decayed | Prevent early wrong collapse |
| D | gumbel | `1.0 -> 0.25` over `300` | small decayed | Only if deterministic gates are informative but training stalls |

Do not run a broad sweep. The goal is to understand the relaxation failure mode
or find a compact positive.

Selection criteria:

- fast-gate normal/operand/pair/calc;
- hard learned pair exact;
- private all-pair operand/pair/calc;
- full-enum learned-minus-true and learned-minus-best gaps;
- hard learned-best fraction;
- final temperature and entropy;
- exact final weights for aux/local/expected/anchor objectives.

## Stage 2: Hard-Only Answer Retention

If Stage 1 reaches a checkpoint with fast-gate operand/pair/calc at or above
`0.90`, continue from the first qualifying checkpoint and the best qualifying
checkpoint with the relaxation fully off:

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

This is the retention claim. Do not call Stage 1 alone a solved result.

Required retention evidence:

```text
relaxed calculator objective exactly inactive
aux/direct supervision exactly 0.0
local target exactly 0.0
expected answer-loss weight exactly 0.0
anchor exactly 0.0
semantic decoder delta 0.0
canonical and private protocol metrics near exact
full-enum learned-minus-true/best gaps near 0.0
learned-best fraction near 1.0
```

## Stage 3: Upstream-Open Branch Only If Needed

Do not start here.

Run an upstream-open strict branch only if Stage 0 shows useful gradient
alignment but Stage 1 frozen-upstream training stalls below the protocol gate.
That pattern would suggest optimization or readout geometry, not target
identifiability, is the blocker.

Use conservative upstream movement:

```text
freeze_upstream_encoder=false
input_proj_lr=0.03
upstream_lr=0.00003 or 0.0001
snapshot_every=25
checkpoint_every=25
```

Require dense checkpoint diagnostics and parameter-delta reporting. If this
branch passes, still run Stage 2 relaxation-off retention.

## Required Diagnostics

For selected checkpoints, run:

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
- hard learned operand exact;
- hard learned pair exact;
- hard learned calculator-result accuracy;
- private all-pair answer/operand/pair/calc;
- full-enum learned NLL, true NLL, and best NLL;
- learned-minus-true and learned-minus-best gaps;
- learned-best and true-best fractions;
- trainable parameter groups;
- semantic decoder parameter delta;
- upstream/input-proj parameter deltas;
- final relaxation temperature;
- exact final weights for aux, local target, expected answer loss, entropy, and
  anchors.

## Decision Criteria

A useful positive:

```text
The relaxed hard-forward answer-loss branch trains hard learned actions
materially above the expected-loss negative and reaches at least partial
protocol quality without hard-best CE or true-operand labels.
```

A strong positive:

```text
Stage 1 reaches near-exact hard learned operand/pair/calc, and Stage 2 retains
near-exact canonical/private/full-enum protocol metrics after the relaxation is
fully off.
```

A useful negative:

```text
The Stage 0 gradient gate shows the relaxed answer-loss gradient does not
increase best-pair probability, or Stage 1 repeats the expected-loss failure:
answer/relaxed loss improves while hard learned actions collapse to wrong
pairs.
```

If negative, do not broaden random sweeps. The next likely move would be to
change parameterization rather than objective, such as a joint-pair relaxation
compatible with `operand_spans`, or to explicitly study why the soft gradient
prefers wrong basins despite the full-enum argmin being sharp.

## Reporting Contract

When complete, update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/<date>-gumbel-concrete-interface-bridge.md
```

Include:

- claim tested;
- code changes;
- exact commands;
- run paths;
- Stage 0 gradient-alignment table;
- Stage 1 ladder table;
- Stage 2 retention table if run;
- selected checkpoint diagnostics;
- exact final objective weights;
- parameter movement summary;
- comparison to the hard-best local-target positive and expected-loss negative;
- go/no-go recommendation.

When fully completed, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

then commit and push, following `CLAUDE.md`.
