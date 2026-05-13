# Phase 7 Second Task: Natural Joint-Pair Stage 1 Result Discovery And Retention Gate

## Mission

Run the first decisive natural `0..19` learned-interface test now that the
joint-pair result-group bridge exists.

The central question is:

```text
Can answer loss train a joint 20 x 20 calculator-query policy to make hard
calculator calls whose results are correct for natural addition, without true
operand labels, oracle operands during bridge training, hard-best pair CE,
expected answer-loss enumeration, or semantic decoder movement?
```

Exact true pair recovery is diagnostic only. In natural sum-only addition,
multiple calculator calls share the same correct result. The primary target is
therefore hard learned calculator-result accuracy, result-equivalent pair
accuracy, and full-enum learned-result alignment.

## Why This Is The Next Best Task

Helpful findings to preserve:

- Phase 4 showed the architecture can carry a real learned calculator-query
  protocol when the target makes operand identity identifiable.
- Phase 5 showed answer-only continuations can preserve or complete a partial
  protocol, but plain no-handoff answer-only training did not discover one from
  scratch.
- Phase 6 showed the strongest result so far: deterministic hard-forward /
  soft-backward Concrete answer-loss training discovered and retained an
  identifiable `sum_left_operand` hard calculator protocol with no true operand
  CE, no hard-best CE, no oracle operands during bridge training, and semantic
  decoder delta `0.0`.
- Phase 6 replicated that deterministic Concrete positive across effective
  seeds `2`, `4`, and `5`, and relaxation-off answer-only retention completed
  all selected protocols.
- The natural product-decoder gate passed after switching the sum-only answer
  decoder to product interaction: oracle-at-eval `1.000`, full-enum
  best-result group true sum `1.000`, injection-zero near chance, semantic
  decoder delta `0.0`.
- The Phase 6 closure diagnostic isolated the natural failure: the answer
  landscape is result-identifiable but pair-underidentified. The true result
  group got about `0.9999` soft target probability, while true pair probability
  was only about `0.0975` and same-true-sum near-best pairs averaged about
  `13.35`.
- The first Phase 7 task removed the immediate implementation blocker:
  `operand_spans + joint_pair + gumbel_concrete_interface` now passes tests and
  sends nonzero answer-loss gradient into `calculator_hook.pair_proj` while
  semantic/upstream gradients stay `0.0`.

Less helpful directions right now:

- More oracle-only decoder/readout reruns after a quick regression gate. They
  are wiring checks, not learned calculator use.
- Repeating natural independent-head deterministic Concrete with small
  schedule changes. That branch already failed after the product decoder gate,
  and the failure mechanism is now understood.
- More exact expected-answer-loss optimization over independent heads. Prior
  branches made expected loss look better while hard learned actions stayed
  wrong.
- More hard-best local-target teaching in the identifiable setting. It is now
  a useful control, not the frontier.
- Literal stochastic Gumbel sampling before the existing instability is
  isolated.
- Scaling to `operand_max=99` before natural `0..19` result-level calculator
  use works.

The fastest honest path is:

```text
joint pair result-group bridge -> hard learned result metric -> relaxation-off
retention if it passes -> seed replication only after retention works.
```

## Read First

Read these before editing or running:

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
aiAgentWorkHistory/phase7/2026-05-12-joint-pair-result-group-bridge-gate.md
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

## Fixed Setup

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
calculator_action_head=joint_pair
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

Use this semantic decoder checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

Use this run root:

```text
runs/2026-05-12_phase7_joint_pair_stage1_result_discovery
```

## Critical Guardrails

Forbidden for training:

- true operand CE;
- true sum CE outside the normal answer target;
- hard-best pair CE;
- hard-best result CE;
- soft targets distilled from full-enum answer losses;
- expected-answer-loss objective;
- oracle operands during bridge training;
- semantic decoder movement.

Allowed for diagnostics only:

- true operands;
- true sums;
- oracle-at-eval;
- forced-zero, injection-zero, forced-random, and forced-result sweeps;
- full-enum best pair/result groups.

Do not proceed to seeds `4` and `5`, upstream-open training, or `operand_max=99`
unless seed `2` first gives a meaningful hard result-level positive.

## Stage 0: Metric And Regression Gate

Before the long Stage 1 run, make sure the task can distinguish three outcomes:

1. soft induced result distribution learns the true result and hard argmax also
   improves;
2. soft induced result distribution learns the true result but hard argmax does
   not harden;
3. neither soft nor hard result-level behavior improves.

If the current logs do not already expose them, add these joint-pair relaxed
metrics to `scripts/overfit_one_batch.py`:

```text
relaxed_calculator_true_result_probability
relaxed_calculator_argmax_result_accuracy
relaxed_calculator_top3_result_accuracy
relaxed_calculator_hard_learned_calc_accuracy
relaxed_calculator_result_entropy
relaxed_calculator_effective_results
relaxed_calculator_pair_entropy
relaxed_calculator_effective_pairs
```

Add focused test coverage if metric code changes.

Then run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Run a quick oracle/readout regression check on the Phase 6 product decoder.
This is a gate only, not a research result:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --oracle --output-dir runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage0_oracle_gate
```

Run the all-400 full-enum result-landscape regression gate:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage0_full_enum_gate
```

Gate:

```text
oracle-at-eval exact >= 0.99
best_result_group_matches_true_sum_fraction >= 0.99
mean_soft_target_true_result_group_probability >= 0.99
mean_soft_target_true_pair_probability remains broad/low; this is expected
semantic decoder delta = 0.0
```

If this gate fails, stop and fix the product decoder/readout regression. Do not
interpret Stage 1 training.

## Stage 1: Strict Seed-2 Joint-Pair Result-Group Bridge

Run one decisive seed before replication:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 600 --batch-size 400 --eval-samples 400 --operand-max 19 --calculator-operand-vocab-size 20 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --answer-format sum --calculator-output-format sum --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --calculator-estimator gumbel_concrete_interface --calculator-action-head joint_pair --calculator-read-position operand_spans --calculator-read-span-width 2 --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 0.5 --relaxed-calculator-temperature-decay-steps 600 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --snapshot-every 25 --snapshot-samples 400 --checkpoint-every 25 --log-every 25 --run-root runs/2026-05-12_phase7_joint_pair_stage1_result_discovery/stage1_seed2_primary
```

Selection rule:

- Select by hard learned calculator-result accuracy first.
- Break ties with full-enum learned-result best fraction and
  learned-result-minus-best-result gap.
- Do not select by normal answer exact alone.
- Pair exact is diagnostic only.

Run canonical diagnostics on the best checkpoint and on any first-gate
checkpoint:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_calculator_protocol.py --checkpoint <selected_checkpoint.pt> --digits 2 --answer-format sum --samples 400 --operand-max 19 --calculator-output-format sum --output-dir <selected_checkpoint_dir>/canonical_diagnostic

PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint <selected_checkpoint.pt> --exhaustive-grid --samples 400 --batch-size 40 --digits 2 --answer-format sum --calculator-output-format sum --operand-max 19 --temperature 0.25 --output-root <selected_checkpoint_dir>/full_enum_diagnostic
```

Stage 1 success gate:

```text
canonical calculator_result_accuracy >= 0.90
canonical result_equivalent_pair_accuracy >= 0.90
full-enum learned_result_best_fraction >= 0.90
mean_learned_result_minus_best_result_gap <= 0.10
oracle-at-eval remains near 1.0
injection-zero and forced-random remain near chance
semantic decoder delta = 0.0
upstream delta = 0.0
aux/direct/expected/local/anchor weights = 0.0
```

Stage 1 near-pass:

```text
canonical calculator_result_accuracy >= 0.50
or relaxed_calculator_true_result_probability is high and rising
```

If Stage 1 is a near-pass but hard argmax does not harden, do not jump to
result-space yet. First diagnose the soft-to-hard handoff:

- compare soft true-result probability vs hard result accuracy over snapshots;
- inspect result entropy and pair entropy;
- try at most one follow-up handoff branch with lower final temperature or a
  short hard-only continuation, clearly labeled as a handoff stabilization.

If Stage 1 stays near the old independent-head negative range, roughly
`0.11-0.14` hard result accuracy with large learned-result gap, record a
joint-pair Stage 1 negative and move next to Track B result-space interface or
Track C canonical symmetry breaker. Do not run seeds `4` and `5`.

## Stage 2: Relaxation-Off Hard Joint-Pair Retention

Run Stage 2 only if Stage 1 passes or strongly near-passes.

The repo may still need one small retention plumbing change: allow
`calculator_action_head=joint_pair` with a hard answer-only estimator such as
`adaptive_interface` when all adaptive/local/expected/relaxed objectives are
inactive. The model forward path already has a hard joint-pair argmax branch;
the CLI validation should not force the relaxed bridge during retention.

Add a regression test if this validation path changes:

```text
joint_pair + adaptive_interface + frozen semantic decoder + no auxiliary
objectives trains/evaluates through the hard argmax calculator path.
```

Then continue from the selected Stage 1 full checkpoint with:

```text
calculator_estimator=adaptive_interface
calculator_action_head=joint_pair
semantic_decoder_checkpoint=<selected_stage1_checkpoint.pt>
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux/adaptive/local/expected/relaxed-entropy/anchor weights all 0.0
```

Retention success gate:

```text
final canonical calculator_result_accuracy >= Stage 1 selected accuracy - 0.05
final full-enum learned_result_best_fraction remains >= 0.90
mean_learned_result_minus_best_result_gap remains <= 0.10
injection-zero and forced-random remain near chance
semantic decoder delta = 0.0
upstream delta = 0.0
all discovery-specific objective weights exactly 0.0
```

If retention improves a near-pass into a pass, label it clearly as
`joint_pair_retention_completion`, matching the Phase 6 precedent.

## Stage 3: Replication Only After Retention

Only after seed `2` passes Stage 2, replicate effective seeds `4` and `5` with
the same Stage 1 and Stage 2 protocol.

Do not modify the semantic decoder, open upstream, add local targets, or scale
operand range during replication.

## Interpretation Matrix

Use these labels:

```text
joint_pair_stage1_positive
joint_pair_soft_result_positive_hard_handoff_negative
joint_pair_stage1_negative
joint_pair_retention_positive
joint_pair_retention_negative
```

Interpretation:

- If soft and hard result metrics both rise, the Phase 7 mainline is working.
- If soft result metrics rise but hard metrics do not, the result objective is
  informative and the next blocker is hardening or symmetry breaking.
- If neither soft nor hard result metrics rise, the joint-pair result-group
  bridge is not enough from strict random initialization; move to Track B
  result-space or Track C canonical query symmetry breaking.
- If Stage 1 passes but Stage 2 fails, the bridge can teach natural result
  use, but answer-only hard continuation cannot yet retain it.

## Reporting Contract

Update:

```text
factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase7/<date>-joint-pair-stage1-result-discovery.md
```

Report:

- exact commands;
- run paths;
- selected checkpoints;
- final objective weights;
- trainable parameter groups;
- semantic, upstream, and pair-proj parameter deltas;
- training-curve hard result, soft true-result probability, result entropy, and
  pair entropy;
- canonical diagnostic table;
- full-enum learned-result table;
- comparison to the Phase 6 natural independent-head negative;
- go/no-go recommendation for retention, replication, Track B, or Track C.

When complete, move this task to:

```text
aiAgentProjectTasks/completed/phase7/
```

Then commit and push.
