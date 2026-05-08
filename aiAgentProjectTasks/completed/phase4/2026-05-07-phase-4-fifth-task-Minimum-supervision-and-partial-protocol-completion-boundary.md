# Phase 4 Fifth Task: Minimum Supervision and Partial Protocol Completion Boundary

## Claim

The previous Phase 4 task established a seed-robust positive:

```text
Full operand-span supervision can teach the calculator-query protocol, and
answer loss can retain that protocol after aux_operand_loss_weight is exactly
0.0.
```

This task asks the next sharper question:

```text
Does answer loss merely preserve an already-learned protocol, or can it
complete and stabilize a partially learned calculator-query protocol after the
teacher signal is removed?
```

Do not introduce a new estimator. Do not unfreeze upstream by default. The goal
is to find the minimum direct operand supervision and handoff quality needed for
aux-zero retention.

## Starting Point

Use the validated Stage 0B semantic decoder checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Known robust positive infrastructure:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- strict `answer_decoder` bottleneck
- semantic decoder and upstream encoder frozen
- trainable group limited to `calculator_hook.input_proj`

Known robust positive runs:

- Stage 1 effective seed `2`:
  `runs/2026-05-07_070737_999460_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2`
- Stage 1 effective seed `4`:
  `runs/2026-05-07_070738_192155_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed4`
- Stage 1 effective seed `5`:
  `runs/2026-05-07_070737_995829_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5`
- Corrected earliest Stage 2A retained selections:
  - seed `2`:
    `runs/2026-05-07_092659_995383_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
  - seed `4`:
    `runs/2026-05-07_074429_578037_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed4/final_weights.pt`
  - seed `5`:
    `runs/2026-05-07_092657_329340_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5/final_weights.pt`

Do not rerun oracle-only controls unless checkpoint loading or calculator-output
wiring has changed.

## Experiment Plan

Run a compact boundary ladder before expanding. Use effective seeds `2`, `4`,
and `5` for the primary ladder. Remember the training script reports the
effective two-digit seed as `--seed + 2`, so the CLI seeds are:

| Effective seed | CLI `--seed` |
| ---: | ---: |
| `2` | `0` |
| `4` | `2` |
| `5` | `3` |

### Stage 1A: Short Aux-Only Warm Starts

Train short supervised warm starts and checkpoint densely. Primary step counts:

```text
10, 25, 50, 75, 100
```

If all three seeds retain from `10`, add a smaller ladder:

```text
1, 2, 5
```

If no seed retains from `50`, add intermediate or longer points:

```text
125, 150
```

Template command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps <stage1_steps> \
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
  --seed <cli_seed> \
  --snapshot-every 5 \
  --checkpoint-every 5 \
  --snapshot-samples 256 \
  --log-every 5
```

For each Stage 1A run, record:

- final checkpoint;
- first checkpoint with operand exact `>= 0.25`;
- first checkpoint with operand exact `>= 0.50`;
- first checkpoint with operand exact `>= 0.75`;
- first checkpoint with operand exact `>= 0.90`;
- first checkpoint with operand exact `>= 0.95`;
- first checkpoint with operand exact `== 1.0`, if any.

### Stage 1B: Decayed-Aux Curricula

Only after Stage 1A shows the rough boundary, test whether a decay schedule can
reduce the amount of direct supervision without creating an ambiguous selected
checkpoint.

Recommended compact ladder:

```text
--steps 300 with aux decayed to 0 by 25, 50, 100
```

Use:

```bash
--aux-operand-loss-weight 1.0
--aux-operand-loss-decay-steps <25|50|100>
--answer-loss-weight 1.0
```

Important: selected checkpoints for a retention claim must have
`final_aux_operand_loss_weight == 0.0` or selected-step aux weight exactly
`0.0`. If the script only records final aux weight, use a final checkpoint after
the decay is complete.

### Stage 2: Aux-Zero Retention From Partial Handoffs

For each seed and selected Stage 1A handoff checkpoint, run aux-zero retention:

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
  --semantic-decoder-checkpoint <stage1_partial_checkpoint> \
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
  --seed <same_cli_seed> \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

Start with at most these Stage 2 handoffs per seed:

```text
lowest available >=0.25, >=0.50, >=0.75, >=0.90, and >=0.95
```

If a lower-quality handoff unexpectedly recovers to `>=0.95`, add one lower
handoff for that seed. If all handoffs below `0.90` fail, stop expanding and
focus diagnostics on the success/failure boundary.

## Fast Gates

For every Stage 1 and Stage 2 run, report:

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

## Diagnostics

Run full diagnostics only on:

- each seed's lowest-supervision retained aux-zero checkpoint;
- the nearest failed checkpoint below that retained boundary;
- the best decayed-aux curriculum checkpoint, if Stage 1B is run.

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

## Success Criteria

A strong positive result:

- at least two seeds retain aux-zero operand exact `>= 0.95` from a handoff that
  was materially below perfect, ideally Stage 1 operand exact `<= 0.75`;
- selected retained checkpoints have `final_aux_operand_loss_weight == 0.0`;
- injection-zero and forced-random remain near chance;
- oracle-at-eval remains `>= 0.95`;
- private all-pair operand exact and calculator-result accuracy remain
  `>= 0.95`;
- full-enum learned-minus-true and learned-minus-best gaps stay near `0.0`.

A useful boundary result:

- handoffs above a threshold, such as `>=0.90`, retain, while lower-quality
  handoffs do not;
- or short aux-only warm starts below some step count never cross a meaningful
  Stage 1 protocol quality threshold;
- or decayed-aux curricula retain only when aux remains active until a measured
  protocol quality threshold is reached.

## Interpretation Rules

- Do not call answer exact alone a success. Addition-like shortcuts and
  answer-decoder recovery are not learned calculator use.
- A checkpoint counts as retained calculator use only if learned operand/pair
  exact, calculator-result accuracy, private all-pair decoding, and full-enum
  gaps agree.
- If a partially learned handoff recovers under aux-zero training, call that
  "answer-loss protocol completion" only if operand/pair exact and
  calculator-result accuracy improve, not merely final answer exact.
- Keep oracle-only controls as wiring checks, not claims.

## Go / No-Go

Go to upstream unfreezing only after the reduced-supervision boundary is known.

Go to identifiable multi-operation or `sum_and_difference` targets if
reduced-supervision retention still requires near-perfect Stage 1 protocol
quality.

No-go on new estimators until this minimum-supervision boundary is measured.

## Deliverables

- Stage 1A short-supervision run paths and selected handoff checkpoints.
- Stage 2 aux-zero run paths and selected retained/failure checkpoints.
- Stage 1B decayed-aux run paths, if run.
- Confirmation that every selected retention checkpoint has aux exactly `0.0`.
- Fast-gate metrics for every run.
- Full diagnostics for selected retained checkpoints and nearest failures.
- Direct comparison to the seed-robust full-supervision result from the
  previous task.
- Fact-sheet and work-history updates.
- Move this task to `aiAgentProjectTasks/completed/phase4/` when complete.
- Commit and push.
