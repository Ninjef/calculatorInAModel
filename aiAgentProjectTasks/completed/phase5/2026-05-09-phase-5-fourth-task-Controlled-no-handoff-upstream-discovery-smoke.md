# Phase 5 Fourth Task: Controlled No-Handoff Upstream Discovery Smoke

## Claim

Phase 5 now has cross-seed upstream-assisted completion positives from failed
partial handoffs:

```text
Opening upstream parameters can complete a partially taught calculator-query
protocol after direct operand supervision is removed, but the positive is
checkpoint-selected rather than all-snapshot stable.
```

The next task should make the smallest honest move toward natural training:

```text
Can answer-only training discover the calculator-query protocol without any
Stage 1 supervised interface handoff, while preserving the strict identifiable
Phase 4/5 semantic bottleneck?
```

This is a **no-handoff upstream/interface discovery smoke test**. It is more
ambitious than upstream-assisted completion, but it is still not unrestricted
end-to-end natural training unless the task explicitly reaches the strict
random-upstream branch and passes its oracle-at-eval gate.

## Starting Point

Read first:

```text
CLAUDE.md
aiAgentProjectTasks/2026-05-08-draft-phase-5-overarching-task-Upstream-discovery-after-protocol-teaching.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase5/2026-05-08-upstream-unfreeze-stability-smoke.md
aiAgentWorkHistory/phase5/2026-05-08-upstream-assisted-partial-handoff-completion.md
aiAgentWorkHistory/phase5/2026-05-08-cross-seed-upstream-assisted-completion-replication.md
runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/summary.md
```

Most relevant existing facts:

- Seed `2`, step `55` and seed `5`, step `25` failed frozen-upstream handoffs
  were completed by upstream-open answer-only continuations.
- The seed `5` primary upstream-open run reached exact true protocol at step
  `950`, then drifted mildly by final.
- The optional seed `5` anchor repeat reached exact true protocol at step
  `800`, then drifted more strongly; anchor `0.001` was not a stability fix.
- Dense checkpoints and full diagnostics are mandatory because final
  checkpoints can be worse than transient retained-protocol checkpoints.

Important implementation wrinkle:

```text
scripts/overfit_one_batch.py currently loads the full checkpoint in
load_semantic_decoder_checkpoint(), not only semantic-decoder tensors.
```

Therefore, a run that uses the Stage 0B checkpoint with upstream unfrozen is a
valid **no-handoff** smoke test, but not a strict random-upstream discovery
test. It starts from the Stage 0B oracle-trained full model state with a random
or untrained learned interface head, not from a wholly random upstream.

## Fixed Semantic/Bottleneck Setup

Keep the Phase 4 and Phase 5 identifiable setup fixed unless this task
explicitly says otherwise:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `calculator_estimator=adaptive_interface`
- `freeze_semantic_decoder=true`
- `answer_loss_weight=1.0`
- `aux_operand_loss_weight=0.0`
- `adaptive_interface_loss_weight=0.0`
- `input_proj_anchor_weight=0.0`
- no oracle training
- no new answer format
- no broadened task
- no direct operand labels

Use a new run root:

```text
runs/2026-05-09_phase5_no_handoff_upstream_discovery_smoke
```

Prefer a small runner script over ad hoc shell commands:

```text
scripts/run_phase5_no_handoff_upstream_discovery_smoke.py
```

The runner should write compact `summary.json` and `summary.md` files that
record run paths, selected checkpoints, fast gates, diagnostics, and parameter
deltas.

## Stage 0: Reuse Evidence and Add Loader Clarity

Do not rerun Phase 4/5 boundary or completion experiments.

Record these references in the new summary:

- Stage 0B operand-aware oracle semantic decoder checkpoint:
  `runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
  if present in this repo, otherwise use the absolute path already recorded in
  the Phase 4 fact sheet.
- Seed `2`, step `55` upstream-assisted completion summary.
- Seed `5`, step `25` cross-seed completion summary.

Before running the main smoke, inspect `load_semantic_decoder_checkpoint()` in
`scripts/overfit_one_batch.py` and confirm whether it still loads the full
checkpoint. If so, record that in the summary as:

```text
semantic_decoder_checkpoint_load_scope = "full_model_current_behavior"
```

Optional but recommended implementation improvement:

Add a backward-compatible opt-in load-scope flag:

```text
--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only
```

Default must be `full_model` to preserve existing runs.

For `semantic_decoder_only`, load only tensors required for the frozen
answer-decoder/calculator-output semantic path, such as:

- `answer_offset_emb.*`
- `answer_decoder.*`
- `calculator_hook.output_proj.*`

Do not silently change old behavior. Add focused tests or at least a
`py_compile` plus a tiny load-scope smoke. If this implementation becomes
messy, skip it and run only the no-handoff full-model smoke below.

## Stage 1: Primary No-Handoff Full-Model Smoke

Run answer-only continuation from the Stage 0B checkpoint, **not** from a Stage
1 supervised interface checkpoint.

Interpretation label:

```text
no_handoff_full_model_init
```

This tests whether answer loss can discover the learned calculator-query
protocol from an untrained learned interface head when upstream is allowed to
move. It is not a partial-handoff completion run because no Stage 1 supervised
interface checkpoint is used.

Run at most two seeds:

- CLI seed `0` / effective seed previously associated with seed `2`;
- CLI seed `3` / effective seed previously associated with seed `5`.

Command template:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 2000 \
  --batch-size 64 \
  --eval-samples 512 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-estimator adaptive_interface \
  --semantic-decoder-checkpoint <stage0b_checkpoint> \
  --freeze-semantic-decoder \
  --answer-loss-weight 1.0 \
  --adaptive-interface-loss-weight 0.0 \
  --aux-operand-loss-weight 0.0 \
  --input-proj-anchor-weight 0.0 \
  --input-proj-lr 0.0003 \
  --upstream-lr 0.00003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --seed <0_or_3> \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

If the runner implements `--semantic-decoder-checkpoint-load-scope`, explicitly
pass:

```text
--semantic-decoder-checkpoint-load-scope full_model
```

Do not run more than these two seeds in this task.

## Stage 2: Diagnostics and Checkpoint Selection

For each Stage 1 run, select:

- final checkpoint;
- best checkpoint by learned protocol fast gates;
- first or worst post-recovery drift checkpoint if any checkpoint reaches
  learned operand/pair/calc `>= 0.999` and later drops below `0.999`.

Run full diagnostics on selected checkpoints:

- canonical causal calculator diagnostics;
- private all-pair protocol diagnostics;
- full-enum action-loss diagnostics.

Success criteria for a **no-handoff discovery positive**:

```text
At any selected checkpoint, canonical operand/pair/calc, private operand/pair/calc,
and full-enum learned-minus-true/best gaps agree that the learned actions are
the true operand protocol, with direct teacher weights exactly 0.0.
```

A final checkpoint is not required for the first smoke positive, but if only a
transient checkpoint succeeds, label the result:

```text
checkpoint-selected no-handoff discovery
```

Failure criteria:

- high answer exact but low learned operand/pair/calc;
- private decoding shows a non-identity/private code;
- full-enum learned-minus-true or learned-minus-best gap remains positive;
- oracle-at-eval succeeds but learned actions never improve above the frozen
  failed partial-handoff baselines;
- direct teacher weights are not exactly `0.0`.

## Stage 3: Optional Strict Random-Upstream Probe

Do not run this branch unless Stage 1 has at least one real no-handoff
discovery checkpoint or the load-scope implementation is very small and safe.

Purpose:

```text
Separate "no supervised handoff" from "pretrained upstream representation".
```

Use the new load-scope flag if implemented:

```text
--semantic-decoder-checkpoint-load-scope semantic_decoder_only
```

Then first run a mechanical oracle-at-eval viability gate. The exact mechanics
can be implemented inside the runner, but the rule is:

- If oracle-at-eval exact is below `0.95`, stop this branch. The fixed semantic
  decoder is too coupled to the Stage 0B upstream for this to be a fair
  random-upstream discovery test yet.
- If oracle-at-eval exact is at least `0.95`, run one seed only with the same
  answer-only setup and dense diagnostics.

Label any result from this branch carefully:

- `strict_random_upstream_discovery_positive` only if learned protocol metrics,
  private all-pair decoding, and full-enum gaps all pass.
- `strict_random_upstream_mechanical_failure` if oracle-at-eval fails before
  learned training is meaningful.

Do not add a new estimator, Gumbel relaxation, REINFORCE, or target-prop
objective in this task. If no-handoff answer-only discovery fails, recommend
the next task rather than starting a method sweep.

## Fast Gates to Report

For every new run and dense selected checkpoint, report:

- run path;
- selected checkpoint path;
- normal exact;
- injection-zero exact;
- forced-random exact;
- oracle-at-eval exact;
- operand exact;
- pair exact;
- calculator-result accuracy;
- mean A/B entropy and confidence;
- `final_aux_operand_loss_weight`;
- `final_adaptive_interface_loss_weight`;
- `final_input_proj_anchor_weight`;
- `freeze_semantic_decoder`;
- `freeze_upstream_encoder`;
- semantic checkpoint load scope;
- trainable parameter groups;
- parameter-delta summary against the starting checkpoint, including upstream,
  `calculator_hook.input_proj`, and semantic decoder movement.

## Interpretation Rules

- Do not call answer exact alone success.
- Do not report oracle-at-eval success as progress; it is a wiring gate.
- Do not call a full-model Stage 0B initialization run "from scratch"; call it
  `no-handoff full-model initialization`.
- Do not call a semantic-decoder-only/random-upstream run meaningful unless
  oracle-at-eval first verifies the fixed decoder can still use the calculator
  path.
- Do not turn this into a broad LR/seed sweep. The win condition is a clean,
  well-diagnosed smoke result, positive or negative.

## Expected Deliverables

- New runner script, preferably:
  `scripts/run_phase5_no_handoff_upstream_discovery_smoke.py`
- Run root with:
  - `summary.json`
  - `summary.md`
  - run logs
  - selected checkpoint diagnostics
- Updated:
  - `factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md`
  - `aiAgentWorkHistory/phase5/<date>-no-handoff-upstream-discovery-smoke.md`
- Move this task file to `aiAgentProjectTasks/completed/` only if the task is
  fully completed.
- Commit and push.

## Recommendation After This Task

If Stage 1 finds a true-protocol checkpoint:

```text
Phase 5 has its first no-handoff answer-only discovery positive. Next work
should focus on stability, narrower unfreeze controls, and whether strict
random-upstream discovery is mechanically viable.
```

If Stage 1 fails cleanly:

```text
Do not broaden into a random sweep. Move next to a single local-target /
full-enum target-prop style objective or a Gumbel-Softmax estimator, while
keeping the same strict identifiable setup and diagnostics.
```

