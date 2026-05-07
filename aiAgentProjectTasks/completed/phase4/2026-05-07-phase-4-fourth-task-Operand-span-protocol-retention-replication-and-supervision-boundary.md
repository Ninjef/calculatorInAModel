# Phase 4 Fourth Task: Operand-Span Protocol Retention Replication and Supervision Boundary

## Claim

The previous one-seed Phase 4 result showed that a supervised
`operand_spans` interface warm start can teach the learned calculator-query
protocol, and that answer loss can retain it when direct operand supervision is
exactly removed.

This task asks the next signal-rich question:

```text
Is aux-zero retained calculator use robust across seeds, and how much supervised
warm start is actually required before answer loss can preserve the protocol?
```

This task should not introduce a new estimator or unfreeze upstream by default.
The goal is replication and boundary-finding around the first positive learned
interface result.

## Starting Point

Use the validated Stage 0B semantic decoder checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Known infrastructure facts:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- strict `answer_decoder` bottleneck
- semantic decoder and upstream encoder frozen
- trainable group limited to `calculator_hook.input_proj`

Known one-seed positive:

- Stage 1 run:
  `runs/2026-05-06_192430_233405_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed3`
- Stage 2 run:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3`
- Stage 2 selected checkpoint had `final_aux_operand_loss_weight=0.0`,
  operand exact `1.0`, private all-pair exact `1.0`, calculator-result accuracy
  `1.0`, and full-enum learned-minus-true gap `0.0`.

Do not rerun oracle-only controls unless checkpoint loading or calculator-output
wiring has changed.

## Experiment Plan

Run a compact replication set first. Suggested logical seeds are `2`, `4`, and
`5` because the training script reports the effective two-digit run seed as
`--seed + 2`. The previous successful effective seed was `3` from `--seed 1`.

### Stage 1: Replicated Aux-Only Warm Starts

For each seed, train only the input projection with direct operand supervision:

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
  --semantic-decoder-checkpoint /Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt \
  --freeze-semantic-decoder \
  --freeze-upstream-encoder \
  --answer-loss-weight 0.0 \
  --adaptive-interface-loss-weight 0.0 \
  --aux-operand-loss-weight 1.0 \
  --aux-operand-loss-decay-steps 0 \
  --input-proj-lr 0.03 \
  --upstream-lr 0.003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --seed <seed> \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

For each seed, identify:

- the first checkpoint with operand exact `>= 0.95` and calculator-result
  accuracy `>= 0.95`;
- the first checkpoint with operand exact `== 1.0`, if any;
- the final checkpoint.

Stage 1 fast gate:

- normal exact `>= 0.95`;
- operand exact `>= 0.95`;
- calculator-result accuracy `>= 0.95`;
- injection-zero and forced-random near chance;
- trainable parameter groups limited to `calculator_hook.input_proj`.

If a seed cannot clear Stage 1, diagnose that seed before running Stage 2 from
it. A Stage 2 retention failure is not meaningful if Stage 1 never learned the
protocol.

### Stage 2A: Retention From Early Handoff

For each seed that clears Stage 1, start from the earliest checkpoint that
passes the Stage 1 gate:

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
  --semantic-decoder-checkpoint <earliest-stage1-gated-checkpoint> \
  --freeze-semantic-decoder \
  --freeze-upstream-encoder \
  --answer-loss-weight 1.0 \
  --adaptive-interface-loss-weight 0.0 \
  --aux-operand-loss-weight 0.0 \
  --input-proj-anchor-weight 0.0 \
  --input-proj-lr 0.0003 \
  --upstream-lr 0.0003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --seed <same-seed> \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

This is the most important boundary test. It asks whether answer loss can
preserve a just-learned protocol, or whether the prior result required a
heavily overtrained, high-confidence Stage 1 interface.

### Stage 2B: Retention From Final Handoff

For each seed, also run Stage 2 from the final Stage 1 checkpoint. This tests
the strongest plausible handoff and should replicate the previous positive if
the result is robust.

Use the same command as Stage 2A, replacing the checkpoint with the final Stage
1 weights.

### Optional Stage 2C: Retention Stress Test

Only after at least two seeds retain under Stage 2A or 2B, run one stress test
on the best retained seed:

- longer aux-zero run: `--steps 3000`; or
- higher retention LR: `--input-proj-lr 0.001`; or
- both, if time is abundant.

Keep `aux_operand_loss_weight=0.0` exactly. Do not introduce anchors,
upstream unfreezing, or adaptive-interface loss in this task.

## Diagnostics

Run full diagnostics only on:

- each seed's best retained aux-zero checkpoint;
- one earliest-handoff failure, if any;
- the stress-test checkpoint, if run.

Canonical diagnostics:

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

## Metrics To Report

For every Stage 1 and Stage 2 run:

- run path;
- selected checkpoint path;
- effective seed and CLI `--seed`;
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
- freeze settings;
- trainable parameter groups.

For Stage 2 selections, also report:

- private all-pair exact;
- private all-pair operand and pair exact;
- private calculator-result accuracy;
- learned A/B affine mapping exactness;
- full-enum learned-best fraction;
- full-enum true-best fraction;
- full-enum learned-minus-true gap;
- full-enum learned-minus-best gap.

## Success Criteria

A strong positive result:

- at least two seeds clear Stage 1;
- at least two aux-zero Stage 2 checkpoints retain operand exact `>= 0.95`;
- selected Stage 2 checkpoints have `final_aux_operand_loss_weight == 0.0`;
- injection-zero and forced-random remain near chance;
- oracle-at-eval remains `>= 0.95`;
- private all-pair operand exact and calculator-result accuracy remain
  `>= 0.95`;
- full-enum learned-minus-true gap stays near `0.0`.

A useful boundary result:

- early handoff fails but final handoff succeeds, showing retention needs a
  sharper Stage 1 interface;
- or one seed fails Stage 1, showing the warm-start optimization is not yet
  seed-robust.

## Go / No-Go

Go to reduced-supervision curricula only if at least two seeds replicate
aux-zero retention.

Go to upstream unfreezing only if interface-only retention is seed-robust.

No-go on new estimators until this replication and boundary result is known.

## Deliverables

- Stage 1 run paths and selected handoff checkpoints.
- Stage 2A and Stage 2B run paths and selected aux-zero checkpoints.
- Confirmation that every selected Stage 2 checkpoint has aux exactly `0.0`.
- Fast-gate metrics for all runs.
- Diagnostics for selected retained checkpoints and one representative failure,
  if any.
- Direct comparison to the one-seed positive from the previous task.
- Fact-sheet and work-history updates.
- Move this task to `aiAgentProjectTasks/completed/phase4/` when complete.
- Commit and push.
