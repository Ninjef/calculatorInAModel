# Phase 5 Third Task: Cross-Seed Upstream-Assisted Completion Replication

## Claim

Phase 5 now has one strong upstream-assisted completion positive:

```text
From the failed seed 2, Stage 1 step 55 handoff, opening upstream parameters
allowed answer-only continuation to recover the true calculator-query protocol
after direct operand supervision was removed.
```

The next task should test whether that result is seed- and handoff-specific, or
whether it generalizes to another known failed Phase 4 boundary point.

Primary question:

```text
Can upstream-open answer-only continuation complete the known failed seed 5,
Stage 1 step 25 handoff after direct operand supervision is removed?
```

This remains **upstream-assisted partial-handoff completion**, not pure
from-scratch discovery.

## Starting Point

Read first:

```text
CLAUDE.md
aiAgentProjectTasks/2026-05-08-draft-phase-5-overarching-task-Upstream-discovery-after-protocol-teaching.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase4/2026-05-08-boundary-closure-before-phase-wrap.md
aiAgentWorkHistory/phase5/2026-05-08-upstream-assisted-partial-handoff-completion.md
```

Primary failed handoff:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed5/2026-05-07_103539_395143_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5/checkpoint_snapshots/step_00025_weights.pt
```

Known facts for this handoff:

- Effective seed `5`, CLI seed `3`.
- Stage 1 step `25` operand exact `0.078125`.
- Frozen-upstream teacher-zero continuation already failed retention:
  final operand/pair/calculator-result accuracy `0.85546875`.
- Diagnostics agreed this was a partial learned protocol, not retained
  true-operand protocol:
  - canonical operand/pair/calc about `0.828`;
  - private operand/pair/calc about `0.848`;
  - full-enum learned-minus-true/best gaps about `0.349`.

Relevant comparison points:

- Seed `5`, Stage 1 step `30` is the nearest retained upper neighbor, around
  Stage 1 operand exact `0.20` and final frozen-upstream retention around
  `0.980`.
- Seed `2`, Stage 1 step `55` was completed by upstream-open continuation,
  with final canonical `0.9961`, private `0.9975`, and full-enum gaps `0.0`.
- The seed `2` upstream-open run had real transient protocol drift before final
  recovery. Keep dense snapshots and diagnose the drift point here too.

## Fixed Experimental Conditions

Keep the Phase 4 and Phase 5 semantic/bottleneck setup fixed:

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
runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion
```

Prefer a small runner script over ad hoc shell commands, for example:

```text
scripts/run_phase5_cross_seed_upstream_assisted_completion.py
```

It is acceptable to adapt the seed `2` runner from:

```text
scripts/run_phase5_upstream_assisted_partial_handoff_completion.py
```

The runner should write compact `summary.json` and `summary.md` files that
record the source handoff, existing frozen baseline, retained upper-neighbor
reference, new run paths, selected checkpoints, fast gates, diagnostics, and
parameter deltas.

## Experiment Plan

### Stage 0: Reuse Boundary Evidence

Do not rerun broad Phase 4 ladders.

Read and record:

```text
runs/2026-05-08_phase4_boundary_closure/summary.json
runs/2026-05-08_phase4_boundary_closure/stage2/seed5/step25
```

Record in the new summary:

- source Stage 1 step `25` checkpoint path;
- Stage 1 handoff operand exact `0.078125`;
- effective seed and CLI seed;
- frozen-upstream failed continuation path and final fast gates;
- frozen-upstream failed continuation diagnostics;
- nearest retained seed `5`, step `30` reference, if available.

If the existing frozen-upstream step `25` baseline is missing or unreadable,
rerun only that matched frozen condition. Otherwise, reuse it.

### Stage 1: Upstream-Open Completion Attempt

Run answer-only continuation from the failed step `25` handoff, with upstream
trainable by omitting `--freeze-upstream-encoder`.

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
  --semantic-decoder-checkpoint <seed5_stage1_step25_checkpoint> \
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
  --seed 3 \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

Primary comparison:

```text
existing frozen-upstream seed5 step25 failure vs new upstream-open seed5 step25 run
```

### Stage 2: Optional Minimal Repeat

Keep this compact. Run at most one optional repeat.

If upstream-open drifts immediately or ends worse than the frozen baseline, run
one lower-upstream-LR repeat:

```text
--upstream-lr 0.00001
```

If upstream-open nearly succeeds but has clear transient or final drift, run
one checkpoint-relative anchor repeat:

```text
--input-proj-anchor-checkpoint <seed5_stage1_step25_checkpoint>
--input-proj-anchor-weight 0.001
```

Do not run both optional repeats in this task unless the first optional result
is unusable due to a mechanical failure. Do not launch a broad sweep.

### Stage 3: Optional Scratch Scout Decision

Only consider this if the seed `5` replication reaches retained-protocol
quality by final diagnostics.

Do **not** run the scratch scout by default in this task. Instead, write a clear
recommendation for the next task. If recommending the scout, keep it bounded:

- one or two seeds;
- no direct operand supervision;
- strict `sum_left_operand` setup unchanged;
- upstream and `calculator_hook.input_proj` trainable;
- dense snapshots and the same canonical/private/full-enum diagnostics;
- success judged by learned operand/pair/calc, private all-pair decoding, and
  full-enum gaps, not answer exact alone.

The purpose of this branch is to decide whether Phase 5 is ready for a
controlled from-scratch discovery smoke test after cross-seed completion
replicates.

## Fast Gates

For every new run and dense snapshot, report:

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
- parameter-delta summary against the source step `25` handoff, including
  upstream group movement.

Also report the existing frozen-upstream step `25` baseline and retained
step `30` reference in the same table where possible.

## Diagnostics

Run full diagnostics on:

- the existing frozen-upstream step `25` failed baseline, if diagnostic outputs
  are not already available in the boundary-closure run;
- the upstream-open final checkpoint;
- the best upstream-open checkpoint by learned protocol metrics;
- the first or worst obvious post-recovery upstream-open drift snapshot if
  drift occurs.

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

Strong cross-seed upstream-assisted completion positive:

- Frozen-upstream seed `5`, step `25` remains failed by existing baseline
  evidence.
- Upstream-open selected checkpoint reaches retained-protocol quality:
  - learned operand exact, pair exact, and calculator-result accuracy are
    `1.000` or effectively `1.000`;
  - private all-pair operand, pair, and calculator-result accuracy are
    `1.000` or effectively `1.000`;
  - full-enum learned-minus-true and learned-minus-best gaps are `0.0` or
    effectively `0.0`.
- Counterfactual controls still show calculator dependence:
  - injection-zero and forced-random near chance;
  - oracle-at-eval `1.000` or effectively `1.000`.
- Direct teacher weights are exactly removed:
  - `final_aux_operand_loss_weight == 0.0`;
  - `final_adaptive_interface_loss_weight == 0.0`;
  - `final_input_proj_anchor_weight == 0.0`, unless explicitly running the
    optional anchor control.
- Upstream parameters changed measurably.

Useful partial positive:

- Upstream-open materially improves over the frozen seed `5`, step `25`
  baseline but does not reach retained-protocol quality.
- Diagnostics show improvement in true operand/pair/calc metrics and reduced
  full-enum gaps, not just higher answer exact.

Useful negative:

- Upstream-open does not improve over the frozen failed baseline, or it drifts
  into lower true-protocol quality despite high answer exact.
- Parameter deltas show upstream moved, so the negative is not a no-op.

No-go / ambiguous:

- Frozen baseline evidence cannot be reproduced or read.
- Upstream-open appears successful by answer exact but learned-interface,
  private, or full-enum diagnostics disagree.
- Upstream-open appears stable only because upstream parameters did not move.

## Reporting Requirements

Update the project record when done:

- Add the Phase 5 result to `factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md`.
- Add work history under `aiAgentWorkHistory/phase5/`.
- If this task is completed, move this task file to
  `aiAgentProjectTasks/completed/phase5/`.
- Commit and push.

The final report should include:

- exact claim tested;
- exact command or runner used;
- run root;
- source handoff path;
- existing frozen baseline path;
- retained seed `5`, step `30` reference path, if available;
- all selected checkpoint paths;
- fast-gate table;
- diagnostic table;
- parameter-delta summary for upstream-open runs;
- comparison against the frozen failed seed `5`, step `25` baseline, the
  retained seed `5`, step `30` reference, and the seed `2`, step `55`
  upstream-assisted positive;
- recommendation for the next Phase 5 task.

## Next Decision

If seed `5` step `25` also completes under upstream-open continuation, Phase 5
has a cross-seed upstream-assisted completion result. The next task should be a
bounded from-scratch discovery smoke test or a teacher-reduction task that
starts from an even weaker handoff.

If seed `5` improves but does not complete, the next task should test a minimal
anchor or narrower unfreeze control.

If seed `5` fails or causes persistent drift, the next task should add narrower
unfreeze controls before trying additional failed handoffs or scratch discovery.
