# Phase 7 Fourth Task: Natural Result-Space Boundary-Target Learning Signal

## Claim

Phase 7 has now ruled out the most direct result-level deterministic Concrete
bridges from strict initialization:

```text
joint_pair_stage1_negative:
  hard learned calculator-result accuracy peaked at 0.11
  soft true-result probability stayed near broad initial mass

result_space_stage1_negative:
  direct 0..38 result request accuracy peaked at 0.0925
  soft true-result probability moved only 0.02564 -> 0.02920
```

This task should test the next, qualitatively different learning signal:

```text
Can an answer-derived boundary target over calculator result classes teach a
natural 0..19 model-side result request, and does that learned request survive
after the boundary-target objective is removed?
```

This is a target-propagation / local-boundary-target experiment, not another
Concrete schedule sweep. The target is computed by asking the frozen downstream
answer decoder which calculator result class minimizes answer NLL for the
current prompt. It must not use true operands or direct result labels to build
the training target. True sums may be used only afterward for diagnostics and
parity checks.

## Why This Is The Next Best Task

Helpful prior findings:

- Phase 4 showed that a taught calculator protocol can be retained after direct
  supervision is removed.
- Phase 5 showed that upstream movement can preserve or complete an already
  partially taught protocol, but no-handoff answer-only discovery is weak.
- Phase 6 showed that answer-derived local targets can teach an identifiable
  protocol, and deterministic Concrete can retain it once learned.
- Phase 6/7 natural gates showed the product answer decoder and result
  landscape are healthy: the true result group is the full-enum best result
  group at `1.000`, and oracle-at-eval recovery is a wiring regression only.
- Phase 7 joint-pair and direct result-space negatives narrowed the blocker:
  pair underidentification was real, but the deeper problem is that
  strict-init answer loss through the current Concrete bridge does not create a
  strong enough interface-teaching signal.

Less helpful to repeat now:

- Oracle-only or forced-true result runs. They are wiring checks, not progress.
- Small LR, temperature, or entropy sweeps of the failed deterministic Concrete
  joint-pair/result-space setup.
- Stage 2 retention, seed replication, canonical-query symmetry breaking, or
  `operand_max=99` scaling from the failed Phase 7 checkpoints.
- Exact true operand-pair recovery as the primary metric for natural sum-only
  addition.
- The old single-sample REINFORCE recipe as-is. It already connected gradients
  but failed in early tiny settings with a scalar moving baseline; policy
  gradient should come back later with a per-prompt multi-sample or
  leave-one-out baseline if this boundary-target branch cannot retain.

The fastest path to the end goal is therefore:

1. Teach the natural result request with an answer-derived local target.
2. Remove that target completely.
3. Test whether the hard calculator-result protocol remains useful under the
   normal answer path.

If this passes, the project has a natural result-level calculator-use positive
with teacher removal. If it fails, the failure is sharply informative: target
propagation can create the protocol, but the current answer-only continuation
cannot hold it.

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase1/2026-04-29-reinforce-calculator-actions.md
aiAgentWorkHistory/phase6/2026-05-10-matched-local-target-teaching-and-retention-gate.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-interface-diagnostic.md
```

Inspect:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/run_phase6_matched_local_target_teaching.py
tests/test_model.py
```

## Fixed Setup

Use the Phase 7 natural `0..19` result-space setup unless a substage explicitly
says otherwise:

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
calculator_action_head=result_space
calculator_read_position=operand_spans
calculator_read_span_width=2
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
```

Use the Phase 6 product-decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Use this run root:

```text
runs/2026-05-13_phase7_result_space_boundary_target_signal
```

## Part 1: Add A Result Boundary-Target Objective

Add the smallest implementation needed to train `calculator_hook.result_proj`
with an answer-derived target over result classes.

Preferred interface:

```text
--result-boundary-target-loss-weight
--result-boundary-target-mode hard_best_result | soft_result
--result-boundary-target-temperature
--result-boundary-target-min-probability-floor
--result-boundary-target-chunk-size
```

If you reuse existing local-target flags internally, still log the result
boundary-target settings under explicit metric names so future readers do not
confuse this with operand-pair local targets.

Implementation requirements:

- Keep `calculator_action_head=result_space`.
- Keep the hard forward path as a real calculator call via the existing
  canonical result-to-pair mapping:
  `a=min(result, operand_max)`, `b=result-a`.
- For a batch, enumerate candidate result classes `0..38`.
- Score each candidate by running the frozen answer decoder path with the
  candidate forced as the calculator result and computing answer NLL.
- Build the training target from those answer NLLs:
  - `hard_best_result`: CE from `result_logits` to the lowest-NLL result class.
  - `soft_result`: KL/CE from `result_logits` to a softmax over negative
    candidate losses.
- Do not use true operands or true sums to construct the target.
- Allow `answer_loss_weight=0.0` during target teaching, so this stage isolates
  the new learning signal.
- Ensure Stage 1 trainable parameters are only `calculator_hook.result_proj`
  when `freeze_upstream_encoder=true`.

Expected tests:

- result-boundary target construction chooses the forced result with lowest
  answer NLL on a controlled batch;
- gradients flow into `calculator_hook.result_proj`;
- gradients do not flow into semantic decoder or upstream parameters when
  frozen;
- hard-best target CE equals direct CE to the true sum only in a parity test
  where true sums are computed after target construction;
- CLI validation rejects invalid modes, negative weights, and non-positive
  temperatures/chunk sizes.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

## Part 2: Stage 0 Boundary-Target Parity Gate

Before training, run a fixed-batch parity/landscape gate from the Phase 6
product decoder checkpoint.

Report:

- hard-best result equals true sum;
- tie-aware true-result best fraction;
- target true-result probability under `soft_result`;
- target entropy and effective result count;
- learned initial hard result accuracy;
- result-proj gradient L2 from the boundary-target loss;
- semantic decoder and upstream gradient/delta exactly `0.0`.

Pass gate:

```text
hard_best_result_equals_true_sum >= 0.98
tie_aware_true_result_best_fraction >= 0.98
semantic_decoder_delta = 0.0
upstream_delta = 0.0
result_proj_gradient_l2 > 0.0
```

If this fails, stop and debug target construction or decoder loading. Do not
proceed to Stage 1.

## Part 3: Stage 1 Boundary-Target Teaching

Run seed `2` only at first.

Use:

```text
answer_loss_weight=0.0
result_boundary_target_loss_weight=1.0
result_boundary_target_mode=hard_best_result
result_boundary_target_temperature=0.25
result_boundary_target_min_probability_floor=0.0
result_boundary_target_chunk_size=64
input_proj_lr=0.03
steps=300
snapshot_every=25
checkpoint_every=25
snapshot_samples=400
```

The existing training CLI uses `--input-proj-lr` as the calculator-hook
learning-rate flag. It is acceptable to reuse that flag for `result_proj`
parameters, but the run metrics should clearly report that the trainable group
is `calculator_hook.result_proj`.

Suggested command shape after implementation:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 300 \
  --batch-size 400 \
  --eval-samples 400 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --answer-format sum \
  --calculator-output-format sum \
  --calculator-bottleneck-mode answer_decoder \
  --answer-decoder-interaction product \
  --calculator-estimator gumbel_concrete_interface \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --freeze-semantic-decoder \
  --freeze-upstream-encoder \
  --answer-loss-weight 0.0 \
  --aux-operand-loss-weight 0.0 \
  --adaptive-interface-loss-weight 0.0 \
  --expected-answer-loss-weight 0.0 \
  --result-boundary-target-loss-weight 1.0 \
  --result-boundary-target-mode hard_best_result \
  --result-boundary-target-temperature 0.25 \
  --result-boundary-target-min-probability-floor 0.0 \
  --result-boundary-target-chunk-size 64 \
  --relaxed-calculator-temperature 1.0 \
  --relaxed-calculator-final-temperature 1.0 \
  --relaxed-calculator-temperature-decay-steps 0 \
  --relaxed-calculator-mode deterministic \
  --relaxed-calculator-hard-forward \
  --relaxed-calculator-entropy-weight 0.0 \
  --input-proj-anchor-weight 0.0 \
  --input-proj-lr 0.03 \
  --upstream-lr 0.0003 \
  --snapshot-every 25 \
  --snapshot-samples 400 \
  --checkpoint-every 25 \
  --log-every 25 \
  --run-root runs/2026-05-13_phase7_result_space_boundary_target_signal/stage1_seed2_hard_best
```

Select the best checkpoint by:

1. hard learned calculator-result accuracy;
2. full-enum learned-result best fraction;
3. lower learned-result minus best-result NLL gap.

Stage 1 pass gate:

```text
hard learned calculator-result accuracy >= 0.90
full-enum learned-result best fraction >= 0.90
mean learned-result minus best-result gap <= 0.25
semantic decoder movement = 0.0
upstream movement = 0.0
```

If Stage 1 does not reach at least `0.70` hard result accuracy, run only one
optimization rescue before stopping:

```text
input_proj_lr=0.01, steps=600, same target settings
```

Do not start broad sweeps.

## Part 4: Stage 2 Boundary-Target-Off Retention

Run only if Stage 1 passes or strongly near-passes.

Start from the selected Stage 1 checkpoint. Turn the boundary target exactly
off and continue with answer loss only through the learned result-space
calculator path.

Use:

```text
answer_loss_weight=1.0
result_boundary_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=true
steps=300
snapshot_every=25
checkpoint_every=25
```

The purpose is retention after the new teaching signal is removed. Do not call
Stage 2 a discovery result unless the final reported objective weights include:

```text
result_boundary_target_loss_weight=0.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
relaxed_calculator_entropy_weight=0.0
input_proj_anchor_weight=0.0
semantic decoder movement=0.0
```

Stage 2 pass gate:

```text
hard learned calculator-result accuracy >= 0.90
canonical normal exact >= 0.90
full-enum learned-result best fraction >= 0.90
injection-zero exact remains near the established wiring-control floor
forced-random exact remains near the established wiring-control floor
oracle-at-eval exact = 1.0, recorded as wiring regression only
```

If Stage 2 drops below `0.70` result accuracy, label it:

```text
boundary_target_teaches_but_answer_only_does_not_retain
```

and recommend a slower target decay or stability regularization task before
replication.

## Diagnostics Required

For the selected Stage 1 and Stage 2 checkpoints, run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_calculator_protocol.py \
  --checkpoint <checkpoint> \
  --digits 2 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum \
  --calculator-output-format sum \
  --calculator-bottleneck-mode answer_decoder \
  --answer-decoder-interaction product \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --eval-samples 400

PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_full_enum_action_loss_diagnostic.py \
  --checkpoint <checkpoint> \
  --digits 2 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum \
  --calculator-output-format sum \
  --calculator-bottleneck-mode answer_decoder \
  --answer-decoder-interaction product \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --samples 400
```

Also report parameter deltas from the relevant step `0` checkpoint:

- `calculator_hook.result_proj`;
- semantic decoder;
- upstream encoder;
- any other trainable group that unexpectedly moved.

## Interpretation Matrix

If Stage 1 and Stage 2 pass:

```text
Phase 7 has a natural result-level calculator-use positive with an
answer-derived boundary-target learning signal and target-off retention.
Next: replicate seeds 4/5, then decide whether to convert result requests into
a stable canonical pair-query protocol or compare against a stronger
multi-sample policy-gradient signal.
```

If Stage 1 passes but Stage 2 fails:

```text
The boundary target can teach the natural result request, but answer-only
continuation cannot yet retain it. Next: target decay, conservative LR, or a
retention stabilizer. Do not replicate until retention is repaired.
```

If Stage 1 fails:

```text
Either the result boundary-target implementation is wrong, or operand-span
features plus result_proj cannot learn even with an explicit answer-derived
target. After the single LR rescue, pivot to a different signal family:
multi-sample policy gradient with per-prompt baselines, surrogate gradients, or
direct feedback alignment.
```

## Reporting Contract

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/2026-05-13-result-space-boundary-target-learning-signal.md
```

When complete:

```text
mv aiAgentProjectTasks/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md \
   aiAgentProjectTasks/completed/phase7/2026-05-13-phase-7-fourth-task-Natural-result-space-boundary-target-learning-signal.md
```

Then commit and push.

Do not present oracle-at-eval success, forced-true result success, or decoder
usability as progress. The research claim lives or dies on learned hard
calculator-result behavior and retention after the boundary-target objective is
exactly off.
