# Phase 7 Third Task: Natural Result-Space Interface Diagnostic

## Mission

Run the next honest Phase 7 diagnostic after strict joint-pair result-group
training failed:

```text
Can answer loss train a model-side natural `0..38` calculator-result request
when the interface action space exactly matches the variable identified by
natural addition answer loss?
```

This is Track B from the Phase 7 plan. It is intentionally weaker than a
learned arbitrary calculator-query claim because a result-space action can look
like answer-class prediction wrapped around a calculator call. Its value is to
separate two possibilities:

1. natural answer loss cannot currently train even a result-aligned calculator
   request from strict initialization;
2. result-level requests are learnable, and the remaining blocker is mapping a
   learned result request into a stable calculator-query protocol.

The decoder/readout is settled infrastructure. Do not rediscover it.

## Intended Training Sequence

This task is the first step in a larger shift toward new training/interface
approaches:

1. **Now: result-space request training.** Add a result-action head and train
   it with answer loss through deterministic hard-forward / soft-backward
   Concrete. The action is the calculator result class, not an arbitrary pair.
2. **If Stage 1 works: objective-off retention.** Continue from the selected
   checkpoint with hard result requests and every discovery-specific objective
   exactly `0.0`.
3. **If retention works: canonical query symmetry breaker.** Use the learned
   result request to impose one deterministic valid calculator query per
   result, then test whether the model can retain an actual query convention.
4. **If the result-space task fails: change the learning signal.** Do not do
   broad small schedule sweeps. Move to qualitatively different training
   methods such as policy-gradient / REINFORCE-style calculator actions,
   target-propagation or local boundary targets, differentiable surrogate
   gradients, synthetic-gradient/direct-feedback methods, or explicit
   curriculum handoffs with teacher removal.

This ordering matters. Result-space is a diagnostic floor: it tells us whether
the result-level request is learnable at all. Canonical query symmetry breaking
comes only after result-space learning works. New estimator families come only
after a result-aligned action space also fails or proves too unstable.

## Why This Is The Next Best Task

The previous Phase 7 task established:

- The natural product decoder/readout remains usable, but this was not new
  research knowledge.
- The full-enum answer-loss landscape is result-sharp:
  true result-group probability `0.99994`, while true pair probability remains
  broad at about `0.09749`.
- Strict joint-pair result-group bridge training did not meaningfully move
  either soft result probability or hard result accuracy:
  best hard learned calculator-result accuracy was only about `0.11`, and soft
  true-result probability stayed near `0.034 -> 0.036`.
- Semantic decoder and upstream deltas remained exactly `0.0`; only
  `calculator_hook.pair_proj` moved.

That negative says the immediate blocker is not merely independent operand
heads. The next question is whether a result-aligned action head learns at all.
If it does, then Track C should convert result requests into stable calculator
queries. If it does not, the problem is deeper than pair underidentification.

## Read First

Read these before editing:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
aiAgentWorkHistory/phase7/2026-05-12-joint-pair-stage1-result-discovery.md
```

Inspect these implementation surfaces:

```text
src/model.py
src/data.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
tests/test_model.py
```

## Non-Negotiable Guardrail

Do not spend this task proving that the decoder can answer with oracle or
forced-true calculator results. That is already known.

Allowed only if result-space implementation touches the semantic decoder,
answer-decoder interaction, calculator output projection, checkpoint loading,
or forced-result path:

```text
one minimal regression check that the existing frozen decoder still works
```

If such a check is run, report it as a regression check only. It must not be
presented as research progress or as a stage gate that was newly discovered.

## Fixed Natural Setup

Use the Phase 7 natural baseline unless a substage explicitly says otherwise:

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
calculator_read_position=operand_spans
calculator_read_span_width=2
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

Use this frozen semantic decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Use this run root:

```text
runs/2026-05-13_phase7_result_space_interface_diagnostic
```

## Forbidden For Training

- true operand CE;
- true pair CE;
- true sum CE outside the normal answer target;
- hard-best pair CE;
- hard-best result CE;
- expected-answer-loss objective;
- soft targets distilled from full-enum answer losses;
- oracle operands during training;
- semantic decoder movement;
- upstream movement before a frozen-upstream result-space pass.

Allowed for diagnostics only:

- true sums;
- learned-result accuracy;
- canonical mapped-pair result accuracy;
- injection-zero, forced-random, and forced-result sweeps;
- private all-result or all-pair probes.

## Stage 0: Implement Result-Space Interface

Add the smallest coherent result-space action head.

Preferred shape:

```text
calculator_action_head=result_space
```

Implementation requirements:

- Add a result-logit projection from the calculator read representation to
  `calculator_result_vocab_size` classes.
- In the hard forward path, select `result_pred = argmax(result_logits)`.
- Convert `result_pred` to a valid calculator query by deterministic canonical
  mapping inside `0..19`, then feed the real calculator result downstream.
- The canonical query must be a pure function of the predicted result and
  operand range, not of true operands. Suggested mapping:

```text
a = min(result, operand_max)
b = result - a
```

For `result > operand_max`, this gives `(operand_max, result - operand_max)`;
for `result <= operand_max`, it gives `(result, 0)`. This covers all natural
`0..38` sums with operands in `0..19`.

Relaxed backward path:

- For deterministic Concrete, build a soft result distribution directly from
  `softmax(result_logits / temperature)`.
- Use hard-forward / soft-backward:

```text
hard_one_hot(result_pred).detach() + soft_result_probs - soft_result_probs.detach()
```

- Route the resulting calculator-output signal through the frozen semantic
  decoder exactly as the normal `sum` calculator output does.

Trace and metrics:

- Add trace fields for `result_pred`, canonical `a_pred`, canonical `b_pred`,
  result confidence, and result entropy.
- Extend `scripts/overfit_one_batch.py` to log:

```text
relaxed_calculator_true_result_probability
relaxed_calculator_argmax_result_accuracy
relaxed_calculator_top3_result_accuracy
relaxed_calculator_hard_learned_calc_accuracy
relaxed_calculator_result_entropy
relaxed_calculator_effective_results
```

- The hard learned calculator-result metric is primary.
- Pair exact is irrelevant except as a sanity check that the canonical mapping
  is valid.

Add focused tests:

- `result_space` forward produces a valid canonical pair for every result
  class `0..38`.
- hard-forward / soft-backward sends nonzero answer-loss gradient into the
  result projection and zero gradient into frozen semantic/upstream parameters.
- CLI validation permits `result_space + gumbel_concrete_interface` and rejects
  incompatible settings cleanly.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

## Stage 1: Strict Seed-2 Result-Space Discovery

Run one decisive seed before any replication:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 600 --batch-size 400 --eval-samples 400 --operand-max 19 --calculator-operand-vocab-size 20 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --answer-format sum --calculator-output-format sum --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --calculator-estimator gumbel_concrete_interface --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 0.5 --relaxed-calculator-temperature-decay-steps 600 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --snapshot-every 25 --snapshot-samples 400 --checkpoint-every 25 --log-every 25 --run-root runs/2026-05-13_phase7_result_space_interface_diagnostic/stage1_seed2_primary
```

Selection rule:

- Select by hard learned calculator-result accuracy first.
- Break ties by relaxed true-result probability, result entropy, and full-enum
  learned-result gap if available.
- Do not select by normal answer exact alone.

Success gate:

```text
canonical calculator_result_accuracy >= 0.90
canonical result_equivalent_pair_accuracy >= 0.90
relaxed_calculator_true_result_probability high and hardened
injection-zero and forced-random near chance
semantic decoder delta = 0.0
upstream delta = 0.0
aux/direct/expected/local/anchor weights = 0.0
```

Near-pass:

```text
hard calculator_result_accuracy >= 0.50
or soft true-result probability is high and rising while hard argmax lags
```

If Stage 1 is a near-pass with soft-but-not-hard learning, run at most one
short handoff stabilization branch with a lower final temperature or hard-only
continuation. Label it clearly as handoff stabilization, not a new mainline.

If Stage 1 is near the joint-pair negative range, record
`result_space_stage1_negative` and stop. Do not run seeds `4`/`5`.

## Stage 2: Objective-Off Retention

Run Stage 2 only if Stage 1 passes or strongly near-passes.

Continue from the selected Stage 1 full checkpoint with:

```text
calculator_estimator=adaptive_interface or a hard result-space argmax estimator
calculator_action_head=result_space
semantic_decoder_checkpoint=<selected_stage1_checkpoint.pt>
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux/adaptive/local/expected/relaxed-entropy/anchor weights all 0.0
```

If the hard result-space estimator does not exist yet, add the smallest CLI
plumbing needed for hard argmax answer-only continuation. Add regression
coverage that all discovery-specific objectives are off and only the
result-space projection trains.

Retention success:

```text
final canonical calculator_result_accuracy >= Stage 1 selected accuracy - 0.05
injection-zero and forced-random near chance
semantic decoder delta = 0.0
upstream delta = 0.0
all discovery-specific objective weights exactly 0.0
```

## Stage 3: Replication

Only after seed `2` passes Stage 2, replicate effective seeds `4` and `5` with
the same Stage 1 and Stage 2 protocol.

Do not modify the semantic decoder, open upstream, add true result labels, or
scale operand range during replication.

## Interpretation Labels

Use these labels:

```text
result_space_stage1_positive
result_space_soft_positive_hard_handoff_negative
result_space_stage1_negative
result_space_retention_positive
result_space_retention_negative
```

Interpretation:

- A Stage 1 positive means natural answer loss can train a result-aligned
  calculator request. It does not yet prove arbitrary query discovery.
- A Stage 2 positive means answer-only hard continuation can retain the
  result-space calculator request after relaxation is off.
- A result-space positive followed by retention should trigger Track C:
  canonical query symmetry breaking, not immediate scaling to `operand_max=99`
  or upstream-open training.
- A Stage 1 negative means the blocker is deeper than joint-pair
  underidentification, and future work should consider local/target-prop,
  policy-gradient, surrogate-gradient, synthetic-gradient/direct-feedback, or
  more explicit curriculum signals.

## Reporting Contract

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-result-space-interface-diagnostic.md
```

Report:

- exact commands;
- run paths;
- selected checkpoints;
- final objective weights;
- trainable parameter groups;
- semantic, upstream, and result-proj parameter deltas;
- training-curve hard result accuracy, soft true-result probability, result
  entropy, and effective results;
- canonical diagnostic table;
- comparison to the joint-pair Stage 1 negative;
- go/no-go recommendation for retention, replication, Track C, or a new
  estimator family.

When complete, move this task to:

```text
aiAgentProjectTasks/completed/phase7/
```

Then commit and push.
