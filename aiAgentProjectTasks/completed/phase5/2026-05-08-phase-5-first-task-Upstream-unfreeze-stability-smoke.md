# Phase 5 First Task: Upstream-Unfreeze Stability Smoke

## Claim

Phase 4 established a true learned calculator-query protocol, but only in a
frozen readable upstream/interface setting. Before Phase 5 asks upstream
parameters to improve failed handoffs or discover a protocol with less teacher
signal, first test the simpler stability question:

```text
Does allowing upstream parameters to move preserve a known retained
true-operand calculator protocol, or does it cause protocol drift?
```

This task is a stability and transfer-readiness probe. It is not a pure
discovery claim.

## Starting Point

Read first:

```text
aiAgentProjectTasks/2026-05-08-draft-phase-5-overarching-task-Upstream-discovery-after-protocol-teaching.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase4/2026-05-08-boundary-closure-before-phase-wrap.md
```

Use the retained Phase 4 seed `2`, Stage 2 step `60` checkpoint:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Source Stage 1 handoff:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/checkpoint_snapshots/step_00060_weights.pt
```

Known baseline facts:

- Effective seed `2`, CLI seed `0`.
- Stage 1 handoff operand exact `0.640625`.
- Frozen-upstream Stage 2 retained at true-protocol quality.
- Selected diagnostics for the retained checkpoint were exact or effectively
  exact: canonical/private operand, pair, and calculator-result accuracy
  `1.000`; full-enum learned-minus-true and learned-minus-best gaps `0.0`.

## Fixed Experimental Conditions

Keep all Phase 4 semantic and bottleneck choices fixed:

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
- no new estimator
- no new answer format
- no broadened task

Use a new run root:

```text
runs/2026-05-08_phase5_upstream_unfreeze_stability_smoke
```

Prefer a small runner script over ad hoc shell commands, for example:

```text
scripts/run_phase5_upstream_unfreeze_stability_smoke.py
```

The runner should write a compact `summary.json` and `summary.md` that record
run paths, selected checkpoints, fast gates, and diagnostic paths.

## Experiment Plan

### Stage 0: Reconfirm the Starting Checkpoint

Do not rerun Phase 4. Just read the existing summary and verify the checkpoint
exists.

Use:

```text
runs/2026-05-07_phase4_min_supervision_boundary/summary.json
```

Record in the new summary:

- retained checkpoint path;
- source Stage 1 checkpoint path;
- Stage 1 handoff operand exact;
- effective seed and CLI seed;
- prior retained diagnostic summary.

### Stage 1: Matched Frozen-Upstream Continuation Control

Run a matched answer-only continuation from the retained checkpoint while
keeping upstream frozen. This confirms that the task setup and checkpoint
loading still reproduce Phase 4 stability.

Command template:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 1000 \
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
  --semantic-decoder-checkpoint <retained_phase4_step60_checkpoint> \
  --freeze-semantic-decoder \
  --freeze-upstream-encoder \
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
  --seed 0 \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

### Stage 2: Upstream-Open Stability Smoke

Run the same continuation with upstream trainable by omitting
`--freeze-upstream-encoder`.

Keep `calculator_hook.input_proj` trainable. Use a conservative upstream LR:

```text
--input-proj-lr 0.0003
--upstream-lr 0.00003
```

This is the primary task condition.

If the run drifts immediately, do not launch a broad sweep. Instead, record the
drift and optionally run one lower-upstream-LR repeat:

```text
--upstream-lr 0.00001
```

If the run remains perfectly stable and upstream movement appears negligible,
optionally run one higher-upstream-LR repeat:

```text
--upstream-lr 0.0001
```

Keep the task compact. The expected total is two required runs and at most one
optional repeat.

## Fast Gates

For every run and dense snapshot, report:

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
- trainable parameter groups;
- whether upstream parameters actually changed, preferably with a small
  parameter-delta summary.

If the current code does not report upstream parameter deltas, add a small
summary helper rather than inspecting manually. At minimum compare the retained
checkpoint to the final checkpoint for the trainable upstream groups.

## Diagnostics

Run full diagnostics on:

- the frozen-upstream continuation final checkpoint;
- the upstream-open final checkpoint;
- the best upstream-open retained snapshot if the final drifts;
- the first obvious drift snapshot if drift occurs.

Canonical causal diagnostics:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m scripts.run_causal_calculator_protocol_diagnostics \
  --checkpoint <selected-weights.pt> \
  --samples 256 \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --forced-result-sweep \
  --forced-result-batch-size 64 \
  --output-dir <run-dir>/canonical_causal_diagnostics
```

Private protocol:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_private_protocol.py \
  --checkpoint <selected-weights.pt> \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --output-dir <run-dir>/private_protocol_diagnostics
```

Full-enum action loss:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_full_enum_action_loss_diagnostic.py \
  --checkpoint <selected-weights.pt> \
  --samples 128 \
  --batch-size 64 \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --temperature 1.0 \
  --chunk-size 64 \
  --output-root <run-dir>/full_enum_action_loss
```

## Success Criteria

Strong stability positive:

- Upstream-open final or selected checkpoint has `final_aux_operand_loss_weight
  == 0.0`, `final_adaptive_interface_loss_weight == 0.0`, and
  `final_input_proj_anchor_weight == 0.0`.
- Normal exact remains near `1.000`.
- Injection-zero and forced-random remain near chance.
- Oracle-at-eval remains `1.000` or effectively `1.000`.
- Learned operand exact, pair exact, and calculator-result accuracy remain
  `1.000` or effectively `1.000`.
- Private all-pair operand, pair, and calculator-result accuracy remain
  `1.000` or effectively `1.000`.
- Full-enum learned-minus-true and learned-minus-best gaps remain `0.0` or
  effectively `0.0`.
- Upstream parameters changed measurably enough that the run was not a no-op.

Useful negative:

- Frozen-upstream control stays retained, but upstream-open drifts.
- The drift is visible in learned operand/pair/calc metrics, private all-pair
  metrics, or full-enum gaps.
- Counterfactual controls distinguish protocol drift from wiring failure.

No-go / ambiguous:

- Both frozen and upstream-open runs fail, suggesting checkpoint loading or
  task setup changed.
- Upstream-open appears stable only because no upstream parameters moved.
- Answer exact stays high while learned protocol metrics degrade.

## Reporting Requirements

Update the project record when done:

- Add a Phase 5 section or new fact sheet entry summarizing the runs and claim.
- Add work history under `aiAgentWorkHistory/phase5/`.
- If this task is completed, move this task file to
  `aiAgentProjectTasks/completed/phase5/`.

The final report should include:

- exact claim tested;
- exact command or runner used;
- run root;
- all selected checkpoint paths;
- fast-gate table;
- diagnostic table;
- parameter-delta summary for upstream-open runs;
- comparison against the original Phase 4 retained seed `2` step `60`
  checkpoint;
- recommendation for the next Phase 5 task.

## Next Decision

If upstream-open stability is positive, the next task should test
upstream-assisted completion from a known failed handoff, starting with:

```text
seed 2, Stage 1 step 55 handoff, Stage 1 operand exact 0.438
```

If upstream-open stability fails, the next task should add narrower unfreeze
controls or checkpoint-relative anchors before attempting failed-handoff
completion.
