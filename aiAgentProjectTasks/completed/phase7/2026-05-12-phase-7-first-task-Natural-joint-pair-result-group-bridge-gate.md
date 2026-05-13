# Phase 7 First Task: Natural Joint-Pair Result-Group Bridge Gate

## Mission

Start Phase 7 with the smallest decisive test of the current hypothesis:

```text
Natural sum-only answer loss identifies the calculator result, not a unique
operand pair. A learned interface should therefore use an action
parameterization that can put probability mass on the entire same-result group.
```

This task should implement and gate a natural `0..19` joint-pair result-group
bridge before any larger operand range, upstream-open branch, or schedule sweep.

The central question is:

```text
Can a joint 20 x 20 calculator-query policy learn hard calculator calls whose
results are correct for natural addition, with no true operand labels, no
oracle operands during bridge training, no hard-best pair CE, and no semantic
decoder movement?
```

Exact true-pair recovery is diagnostic only. In natural sum-only addition,
`03+07`, `04+06`, `05+05`, and other same-sum queries are answer-equivalent.
The primary learned-interface target is the calculator result produced by the
hard learned call.

## Why This Is The Next Best Task

Helpful findings to preserve:

- Phase 4 proved the architecture can carry a real learned calculator-query
  protocol when the target makes operand identity identifiable.
- Phase 5 showed answer-only continuations can preserve or complete a partial
  protocol, but plain no-handoff answer-only training does not discover one.
- Phase 6 hard-best local targets showed frozen answer-decoder NLL can expose a
  sharp interface target in the identifiable setting.
- Phase 6 deterministic hard-forward / soft-backward Concrete is the strongest
  current result: answer loss itself discovered and retained an identifiable
  hard calculator protocol without true operand CE, hard-best CE, oracle
  operands during bridge training, or semantic decoder movement.
- Phase 6 replicated that deterministic Concrete positive across effective
  seeds `2`, `4`, and `5`, and it survived relaxation-off retention.
- The natural product-decoder gate passed: oracle-at-eval and full-enum
  best-result group were exact enough, while injection-zero and forced-random
  stayed near chance.
- The Phase 6 closure landscape found natural sum-only is result-identifiable
  but pair-underidentified: true result group probability was about `0.9999`,
  while the true pair probability was about `0.0975` and near-best same-sum
  pairs averaged about `13.35`.

Less helpful directions right now:

- More oracle-only decoder/readout reruns after the product gate has passed.
  Those are wiring checks, not learned calculator use.
- Repeating natural deterministic Concrete with independent operand heads and
  only small schedule changes. The current failure is explained by the
  parameterization/objective mismatch, not by a missing minor sweep.
- More exact expected-answer-loss optimization over independent heads. Prior
  expected-cost branches reduced expected loss while hard actions collapsed to
  wrong protocols.
- More hard-best local-target teaching in the identifiable task. It works and
  is now a control, not the frontier.
- Literal stochastic Gumbel training before the known instability/variance
  problem is isolated.
- Scaling to `operand_max=99` before natural `0..19` has a learned
  result-level interface.

The fastest honest path to the end goal is:

```text
match the action parameterization to the natural result-level signal -> test
hard learned result accuracy -> only continue to retention/replication if the
hard calculator result actually improves.
```

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
aiAgentWorkHistory/phase6/2026-05-12-phase-6-closure-landscape-diagnostic.md
```

Inspect:

```text
src/model.py
src/data.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase6_closure_landscape_diagnostic.py
scripts/run_phase6_sum_only_semantic_decoder_gate.py
tests/test_model.py
```

## Current Code Boundary

The repo already has some useful pieces:

- `calculator_action_head=joint_pair` exists.
- `pair_proj` emits `V x V` pair logits and traces `pair_pred`, `a_pred`,
  `b_pred`, `result_pred`, pair confidence, and pair entropy.
- Full-enum diagnostics already understand joint-pair checkpoints and report
  result-group metrics such as learned-result best fraction and result-group
  probabilities.

But the current code does not yet support the desired Phase 7 bridge:

- `scripts/overfit_one_batch.py` rejects
  `calculator_estimator=gumbel_concrete_interface` with
  `calculator_action_head=joint_pair`.
- `calculator_read_position=operand_spans` currently requires independent
  operand heads, even though Phase 4 found span reads were important for
  two-digit operands under frozen upstream representations.
- The `CalculatorHook.forward` joint-pair branch currently uses a hard argmax
  pair directly and does not build a hard-forward / soft-backward result-group
  calculator signal.

This task should remove only the relevant blockers. Do not redesign the whole
training stack.

## Fixed Setup

Unless a substage explicitly says otherwise:

```text
digits=2
operand_max=19
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
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
oracle_train=false
oracle_warmup_steps=0
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
```

Use the Phase 6 product decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

If absent locally, use the exact path recorded in
`factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md`.

Use a new run root:

```text
runs/2026-05-12_phase7_joint_pair_result_group_bridge_gate
```

## Critical Guardrail

Do not rediscover oracle success.

Allowed for training:

- answer loss through a relaxed joint-pair calculator signal;
- pair logits produced by the learned interface;
- deterministic Concrete / softmax relaxation over pair logits;
- hard forward through the real selected calculator pair;
- soft backward through the result distribution induced by summing pair
  probability over all pairs with the same calculator result;
- optional entropy or temperature schedules only if clearly labeled.

Forbidden for training:

- true operand CE;
- true sum CE outside the normal answer target;
- hard-best pair CE;
- hard-best result CE;
- soft targets distilled from full-enum answer losses;
- expected-answer-loss objective;
- oracle operands during bridge training;
- semantic decoder movement.

True operands, true sums, full-enum best groups, and oracle-at-eval may be used
only for diagnostics and interpretation.

## Stage 0: Result-Group Joint-Pair Implementation Gate

Implement the smallest joint-pair relaxed bridge needed for this task.

Preferred behavior:

```text
calculator_estimator=gumbel_concrete_interface
calculator_action_head=joint_pair
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
```

For each example:

1. Produce joint pair logits over `20 x 20` possible calculator calls.
2. Form a soft pair distribution:

```text
p_pair = softmax(pair_logits / temperature)
```

3. Choose the hard forward pair by argmax.
4. Run the real calculator on that hard pair.
5. Form the soft result distribution:

```text
p_result[s] = sum_{a+b=s} p_pair[a,b]
```

6. Use hard-forward / soft-backward signal:

```text
signal = hard_result.detach() + p_result - p_result.detach()
```

For this task, `calculator_output_format=sum` is sufficient. Do not add
`sum_left_operand` joint-pair relaxation unless it falls out naturally.

Support `calculator_read_position=operand_spans` for `joint_pair`, because the
existing two-digit learned-interface positives depended on span reads. A
reasonable minimal shape is:

```text
pair_proj: Linear(2 * calculator_read_span_width * n_embd, operand_vocab^2)
```

where the input is the concatenation of the A operand span representation and
the B operand span representation. Preserve existing `operands` behavior for
older joint-pair tests/checkpoints.

Required tests:

- joint-pair deterministic relaxed path produces a hard result in the forward
  trace and gradient on `calculator_hook.pair_proj`;
- semantic decoder and upstream parameters remain frozen when requested;
- `operand_spans + joint_pair` builds and uses the expected projection shape;
- existing independent-head relaxed calculator tests still pass;
- existing joint full-enum interface behavior still passes.

If this implementation cannot be done cleanly in a small patch, stop after a
one-step gradient gate that proves the blocker and write that up. Do not
replace this with a broad independent-head rerun.

## Stage 1: Natural Decoder And Landscape Regression Gate

Before training the bridge, verify the existing natural product decoder is
still healthy. Do not retrain the oracle decoder unless the checkpoint is
missing or the gate unexpectedly fails.

Required gates on the all-400 natural grid:

```text
oracle-at-eval exact >= 0.99
best-result group matches true sum >= 0.99
injection-zero near chance
forced-random near chance
semantic decoder delta exactly 0.0
```

Also record the underidentification baseline:

```text
true result group soft probability high
true pair soft probability low/broad
same-true-sum near-best pair count high
```

This gate is a wiring and landscape regression check only.

## Stage 2: One-Step Joint-Pair Gradient Gate

Run a cheap one-step or few-step diagnostic from a strict
`semantic_decoder_only`, frozen-upstream initialization before launching a full
training branch.

Report before/after:

- hard learned calculator-result accuracy;
- hard result-equivalent pair accuracy;
- full-enum learned-result best fraction;
- learned-result minus best-result NLL gap;
- pair entropy and effective pair count;
- result entropy and effective result count;
- pair-proj gradient norm;
- semantic decoder gradient/delta;
- upstream gradient/delta.

The useful sign is not necessarily large movement after one step. The gate
should at least prove:

```text
answer loss sends nonzero gradient to pair_proj through result-group mass
while semantic decoder and upstream stay fixed.
```

If the one-step gate is zero, NaN, or only changes semantic/upstream/decoder
parameters, fix the bridge before training.

## Stage 3: Seed-2 Strict Joint-Pair Bridge

Run only effective seed `2` first.

Suggested starting branch:

```text
calculator_estimator=gumbel_concrete_interface
calculator_action_head=joint_pair
calculator_read_position=operand_spans
calculator_read_span_width=2
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
input_proj_lr=0.03
upstream_lr=0.0003
steps=300 to 600
snapshot_every=25 or 50
checkpoint_every=25 or 50
eval_samples=400
```

Despite the name `input_proj_lr`, make sure the trainable interface group for
this branch includes `calculator_hook.pair_proj` and not the frozen
independent-head `input_proj`.

Select checkpoints by learned-interface metrics, not by final answer exact
alone:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- normal answer exact;
- private all-pair result accuracy;
- full-enum learned-result best fraction;
- learned-result minus best-result gap;
- injection-zero and forced-random controls;
- semantic decoder delta.

Do not continue to retention unless the hard learned result metrics materially
beat the Phase 6 independent-head natural negative. As a rough gate:

```text
fast learned result accuracy >= 0.70 is a debug-positive
fast learned result accuracy >= 0.90 is a retention candidate
```

If the best checkpoint stays near the old independent-head result
(`~0.11` to `0.14`), stop and write the negative clearly.

## Stage 4: Relaxation-Off Retention Only If Stage 3 Passes

From the best Stage 3 checkpoint, continue with the real hard calculator path
and all discovery-specific objectives off:

```text
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=true
```

If the code requires changing `calculator_estimator` away from
`gumbel_concrete_interface` for hard-only retention, be explicit about the
exact estimator and verify the forward path uses the hard learned joint pair.

Retention success requires:

- normal answer exact near exact;
- learned calculator-result accuracy near exact;
- result-equivalent pair accuracy near exact;
- private all-pair result accuracy near exact;
- full-enum learned-result best fraction near `1.0`;
- learned-result minus best-result gap near `0.0`;
- injection-zero and forced-random near chance;
- semantic decoder delta `0.0`;
- all direct/discovery objective weights exactly `0.0`.

## Stage 5: Replication Only After Retention

Do not run seeds `4` and `5` unless seed `2` passes Stage 4.

If seed `2` passes, replicate the same bridge and retention stages across
effective seeds:

```text
2, 4, 5
```

Keep dense checkpoints and use the same selection criteria.

## Required Diagnostics

For every selected checkpoint, run or extend canonical diagnostics to report:

- built-in eval exact;
- normal answer exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- oracle-at-eval exact;
- learned operand exact, diagnostic only;
- learned pair exact, diagnostic only;
- result-equivalent pair accuracy;
- learned calculator-result accuracy;
- private all-pair answer exact;
- private all-pair result accuracy;
- full-enum learned result NLL;
- full-enum true result NLL;
- full-enum best result NLL;
- learned-result minus best-result gap;
- learned-result best fraction;
- best-result group true-sum fraction;
- pair entropy and effective pair count;
- result entropy and effective result count;
- trainable parameter groups;
- pair-proj, input-proj, upstream, and semantic decoder parameter deltas;
- exact final objective weights.

## Success Definitions

Useful implementation positive:

```text
Joint-pair hard-forward / soft-backward result-group training is wired, sends
gradient to pair_proj, preserves semantic decoder freeze, and passes all tests.
```

Useful Stage 3 positive:

```text
Without true operand labels or oracle operands, the hard learned calculator
result rises materially above the old independent-head natural negative, and
the full-enum learned-result gap closes.
```

Strong Phase 7 first-task positive:

```text
Seed 2 learns a natural hard calculator-result protocol and retains it after
the relaxation is off, with semantic decoder delta 0.0 and all direct or
discovery objectives exactly off.
```

Very strong positive:

```text
The retained seed-2 result replicates across effective seeds 4 and 5.
```

## Failure Interpretation

If Stage 1 fails:

```text
Do not train. Fix the natural product decoder/readout checkpoint or diagnostic
loading first.
```

If Stage 2 has no useful pair-proj gradient:

```text
The bridge is not actually connecting answer loss to the joint-pair policy.
Fix implementation before training.
```

If Stage 2 has gradient but Stage 3 remains near the old independent-head
negative:

```text
Joint-pair result-group relaxation alone is not enough from this strict
initialization. Prefer a result-space diagnostic or a canonical-query
symmetry-breaker before any larger schedule sweep.
```

If Stage 3 succeeds but Stage 4 fails:

```text
The relaxed bridge can teach a natural result protocol, but hard answer-only
retention cannot yet hold it. Investigate slower handoff, gate-triggered
handoff, or minimal stability regularization.
```

## Reporting Contract

When complete, update or create:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-joint-pair-result-group-bridge-gate.md
```

Create these directories if needed:

```text
aiAgentWorkHistory/phase7/
aiAgentProjectTasks/completed/phase7/
```

The work history and fact sheet update must include:

- claim tested;
- code changes;
- exact commands;
- run paths;
- selected checkpoints;
- Stage 1 landscape gate;
- Stage 2 gradient gate;
- Stage 3 bridge table;
- Stage 4 retention table if run;
- objective weights;
- parameter movement summary;
- comparison to Phase 6 identifiable positive and natural independent-head
  negative;
- go/no-go recommendation.

When fully complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase7/
```

then commit and push.
