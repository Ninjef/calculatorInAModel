# Phase 6 Fourth Task: Strict Local-Target Decay And Minimum Teaching Boundary

## Claim

Phase 6 has now established the important strict-branch positive:

```text
With only the frozen semantic decoder loaded from Stage 0B, the answer-derived
hard-best local target can teach the true calculator-query protocol from the
strict random/new interface initialization, and answer-only continuation
retains that protocol after the local target is exactly 0.0.
```

The next best task is therefore not another proof that the full-strength local
target works. It is a boundary task:

```text
How little answer-derived local teaching is needed in the strict
semantic-decoder-only branch before answer loss can retain or complete the true
calculator-query protocol with the local target exactly 0.0?
```

This is the fastest useful move toward the end goal because the current local
target is highly informative and, under the identifiable task, its hard-best
labels match direct operand labels. The next interpretive crutch to remove is
not the Stage 0B upstream representation; Phase 6 third already removed that.
The next crutch is full-strength, fixed-duration local teaching.

## Why This Is The Next Best Task

Helpful findings to carry forward:

- Phase 4 made the answer target identifiable with `sum_left_operand`.
- Phase 4 showed teacher-zero retention can keep a true calculator-query
  protocol after direct operand supervision is exactly removed.
- Phase 5 showed plain answer-only no-handoff training does not discover the
  protocol, but answer-only continuation can preserve or complete a partially
  taught protocol.
- Phase 6 first showed the answer-derived full-enum target is sharp in the
  identifiable setup: best=true `1.000`, effective pairs about `1.079`, and
  true-pair probability about `0.989`.
- Phase 6 second showed the hard-best local target can replace direct
  true-operand labels in the full-model branch.
- Phase 6 third showed the same recipe succeeds in the stricter
  `semantic_decoder_only` branch, so the result no longer depends on loading
  the oracle-trained Stage 0B upstream representation.

Less helpful to repeat now:

- Oracle-only runs beyond a wiring/parity check. Oracle-at-eval remains a gate,
  not research progress.
- More full-strength constant local-target teaching at
  `local_target_loss_weight=1.0` for `300` steps. That already works.
- More full-model load-scope training. The stricter `semantic_decoder_only`
  branch has passed.
- Broad answer-only no-handoff sweeps. Phase 5 already made that a low-value
  direction.
- Gumbel, joint-pair, sampled-candidate replay, or larger operand ranges before
  establishing the minimum local-teaching boundary in the current clean setup.
- Upstream-open variants as the immediate next task. They are useful later if
  the research question requires upstream movement, but they still leave the
  full local-teacher issue unresolved unless paired with a decay/minimum-signal
  test.

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
aiAgentWorkHistory/phase6/2026-05-11-strict-random-upstream-local-target-discovery.md
```

Inspect:

```text
scripts/run_phase6_strict_random_upstream_local_target.py
scripts/run_phase6_matched_local_target_teaching.py
scripts/overfit_one_batch.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_private_protocol.py
src/model.py
tests/test_model.py
```

## Fixed Setup

Keep the Phase 4/5/6 identifiable strict setup:

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
semantic_decoder_checkpoint_load_scope=semantic_decoder_only for new Stage 1 runs
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
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
runs/2026-05-11_phase6_strict_local_target_decay_boundary
```

## Part 1: Extend The Strict Runner For Decay

Extend:

```text
scripts/run_phase6_strict_random_upstream_local_target.py
```

or create a narrow successor:

```text
scripts/run_phase6_strict_local_target_decay_boundary.py
```

The runner should support:

```text
oracle-wiring-gate
compare-local-target-to-aux
run-decay-ladder
run-minimum-handoff
diagnostics
summarize
```

Required runner changes:

- Thread `--local-target-decay-steps` through to
  `--adaptive-interface-loss-decay-steps`.
- Thread `--local-target-floor` through to
  `--adaptive-interface-loss-floor`; default must be `0.0`.
- Include initial local target weight, decay steps, floor, and final local
  target weight in labels, `commands.jsonl`, `metrics.json` summaries, and
  final tables.
- Keep `semantic_decoder_checkpoint_load_scope=semantic_decoder_only` as the
  default for new Stage 1/decay runs.
- Keep retention and manual checkpoint continuations loading learned
  checkpoints with `semantic_decoder_checkpoint_load_scope=full_model`.
- Preserve the existing parity gate: it should still build the model through
  the same strict load path used by training.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase6_strict_local_target_decay_boundary.py scripts/run_phase6_strict_random_upstream_local_target.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

If adapting the existing strict runner instead of creating a new one, compile
only the files that exist.

Add or update a focused test if command construction is changed. The test
should verify that a decay run passes:

```text
--semantic-decoder-checkpoint-load-scope semantic_decoder_only
--adaptive-interface-loss-decay-steps <N>
--adaptive-interface-loss-floor 0.0
```

## Part 2: Reconfirm Gates Without Repeating Oracle Work

Use the prior strict branch gates if code paths are unchanged:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/oracle_wiring_gate.json
runs/2026-05-11_phase6_strict_random_upstream_local_target/parity_gate.json
```

If runner/model wiring changes, rerun the gates once under the new run root.

Gate A pass criteria:

```text
oracle-at-eval exact near 1.000
injection-zero exact near 0.0
forced-random near chance
semantic decoder delta 0.0
```

Gate B pass criteria:

```text
hard_best_pair_equals_true_pair >= 0.98
abs(local_minus_aux_ce) <= 1e-6
semantic_decoder_grad_l2 == 0.0
semantic_decoder_delta_l2 == 0.0
```

Do not spend additional time on oracle-only diagnostics after these gates pass.

## Part 3: Single-Stage Decayed Local-Target Ladder

Primary question:

```text
Can answer loss retain or complete the protocol when the answer-derived local
target decays to exactly 0.0 during one strict semantic-decoder-only training
run?
```

Use:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
local_target_loss_weight=1.0 initially
local_target_loss_floor=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
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

Run a compact decay ladder, stopping early if a shorter decay clearly passes:

| Branch | Decay steps | Purpose |
| --- | ---: | --- |
| A | `50` | Aggressive minimum-signal attempt |
| B | `75` | Match the first prior protocol gate step |
| C | `100` | Middle branch if `50/75` are unstable |
| D | `150` | Conservative decay before falling back to two-stage handoff |

Do not run more branches unless the boundary is ambiguous, such as one branch
ending just below the gate with improving dense snapshots.

Pass criteria for a decayed branch:

```text
final_local_target_loss_weight == 0.0
aux_operand_loss_weight == 0.0
input_proj_anchor_weight == 0.0
semantic decoder delta == 0.0
canonical operand/pair/calc near 1.0
private operand/pair/calc near 1.0
full-enum learned-minus-true and learned-minus-best gaps near 0.0
learned-best fraction near 1.0
injection-zero near 0.0
forced-random near chance
```

Interpretation:

- If `50` or `75` passes, Phase 6 has a strong minimum-teaching result: a brief
  answer-derived local signal is enough for answer loss to carry the protocol.
- If only `100` or `150` passes, the method is still useful but needs a longer
  local-teacher window.
- If every decayed branch fails while full-strength Stage 1 still passes, the
  blocker is handoff dynamics, not target identifiability. Move to Part 4.

## Part 4: Minimum Two-Stage Handoff Boundary

Run this part only if the single-stage decayed ladder fails or is ambiguous.

Purpose:

```text
Find the shortest local-target-only pretrain checkpoint from the existing
strict branch that answer-only retention can complete after the local target is
set exactly to 0.0.
```

First inspect the existing strict Stage 1 dense snapshots:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage1/semantic_decoder_only_branch_a_frozen_upstream_inlr0.03
```

Run manual retention from earlier checkpoints that did not already receive full
diagnostics, prioritizing:

```text
step_00025_weights.pt
step_00050_weights.pt
step_00075_weights.pt
```

Use:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
upstream_lr=0.00003
steps=1000
snapshot_every=50
checkpoint_every=50
seed=0
```

Interpretation:

- If step `25` or `50` retention completes to exact protocol metrics, the
  local target only needs to kick the interface into the right basin.
- If step `75` is the earliest reliable handoff, that agrees with the prior
  first-gate result and gives a concrete minimum boundary.
- If only step `125` works, the current recipe requires near-exact local-target
  teaching before answer-only retention can safely take over.

## Optional Part 5: Upstream-Open Check Only After A Boundary Passes

Run this only after either Part 3 or Part 4 establishes a useful minimum
teacher boundary.

Purpose:

```text
Check whether allowing upstream parameters to move preserves the best
minimum-teaching result, without making upstream movement the main claim.
```

Use the best passing decay or handoff checkpoint and run one upstream-open
answer-only continuation:

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

Require dense diagnostics and parameter movement reporting. If it drifts, do
not broaden into a sweep in this task; record the drift and recommend a
separate upstream-stability task.

## Required Diagnostics

For selected checkpoints, run:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

At minimum diagnose:

- all final decayed ladder checkpoints;
- the shortest decayed branch that passes the fast gate;
- the nearest failing shorter decayed branch;
- any manual minimum-handoff checkpoint that passes;
- any optional upstream-open checkpoint.

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
- full-enum learned NLL, true NLL, best NLL;
- learned-minus-true and learned-minus-best gaps;
- learned-best fraction;
- true-best fraction;
- initial and final local-target weights;
- decay steps and floor;
- exact final objective weights;
- trainable parameter groups;
- semantic decoder parameter delta;
- input-proj and upstream parameter deltas.

## Success Definitions

### Useful Positive

A single-stage strict branch reaches retained-protocol quality after the local
target decays to exactly `0.0`:

```text
semantic_decoder_only initialization
answer_loss_weight=1.0
final_local_target_loss_weight=0.0
aux_operand_loss_weight=0.0
canonical/private/full-enum protocol metrics near exact
semantic decoder delta 0.0
```

### Stronger Positive

The shortest passing decay is `50` or `75` steps, or answer-only retention
completes from a local-target-only checkpoint before the prior first exact
checkpoint.

### Negative But Useful

Decayed single-stage runs fail, but two-stage handoff from step `75` or `125`
still succeeds. This means the local target can teach a protocol, but the
combined answer/local objective or early handoff dynamics need redesign.

### Stop Condition

If gates pass but all decayed and minimum-handoff branches fail, stop after
documenting the failure. Do not broaden into architecture changes inside this
task. The next task should then consider a softer relaxation, a local-target
schedule redesign, or an explicitly upstream-open stability/control plan.

## Reporting Contract

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-11-strict-local-target-decay-boundary.md
```

The work history should include:

- claim tested;
- why this followed the strict random-upstream positive;
- code changes;
- exact commands;
- run paths;
- gate reuse or rerun details;
- decay ladder table;
- minimum handoff table if run;
- selected checkpoints;
- full diagnostics table;
- exact final objective weights;
- parameter movement summary;
- comparison to Phase 6 full-strength strict branch;
- go/no-go recommendation.

When complete, move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Then commit and push, following `CLAUDE.md`.
