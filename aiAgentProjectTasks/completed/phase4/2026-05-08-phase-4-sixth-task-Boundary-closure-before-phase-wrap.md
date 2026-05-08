# Phase 4 Sixth Task: Boundary Closure Before Phase Wrap

## Claim

The previous task established the important Phase 4 positive:

```text
Answer loss can complete and stabilize a partially learned calculator-query
protocol after direct operand supervision is removed.
```

It also left one boundary loose:

```text
The minimum handoff quality is seed-dependent, and seed 5 retained even from
the one extra below-gate handoff.
```

This task is deliberately narrow. Its purpose is to sharpen the partial-handoff
boundary enough to decide whether Phase 4 is complete, or whether the boundary
hides a confound that needs another Phase 4 task.

Do not introduce new estimators. Do not unfreeze upstream. Do not add new answer
formats. This is a closure probe, not a new research direction.

## Starting Point

Use the Phase 4 boundary runner and run root from the previous task:

```text
scripts/run_phase4_min_supervision_boundary.py
runs/2026-05-07_phase4_min_supervision_boundary
```

Validated Stage 0B semantic decoder checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Shared Phase 4 infrastructure:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- strict `answer_decoder` bottleneck
- semantic decoder frozen
- upstream encoder frozen
- trainable parameters limited to `calculator_hook.input_proj`
- Stage 2 handoffs must have `aux_operand_loss_weight=0.0`
- selected retention checkpoints must have `final_aux_operand_loss_weight=0.0`

Known boundary from the previous task:

| Effective seed | Failed lower handoff | Lowest retained handoff | Status |
| ---: | ---: | ---: | --- |
| `2` | step `25`, Stage 1 operand `0.320`, final `0.773` | step `60`, Stage 1 operand `0.641`, final `1.000` | gap too wide |
| `4` | step `30`, Stage 1 operand `0.188`, final `0.699` | step `35`, Stage 1 operand `0.363`, final `0.980` | reasonably bracketed |
| `5` | not measured | step `30`, Stage 1 operand `0.230`, final `0.980` | lower failure still needed |

## Experiment Plan

### Stage 1: Inspect Existing Dense Stage 1A Snapshots

Do not rerun Stage 1A unless the existing dense snapshots are missing.

For seeds `2`, `4`, and `5`, inspect:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage1a/seed*/.../diagnostic_snapshots.csv
```

Select only the smallest useful set of additional Stage 2 handoffs:

- Seed `2`: choose one or two handoffs between step `25` and step `60`, ideally
  around operand exact `0.40` to `0.55`.
- Seed `5`: choose one or two handoffs below step `30`, ideally around operand
  exact `0.10` to `0.22`, to find a failed lower neighbor.
- Seed `4`: run at most one midpoint only if an existing snapshot gives a clear
  handoff between operand exact `0.19` and `0.36`. If no useful midpoint exists,
  skip seed `4`.

Keep the expansion compact. A good target is `3` to `5` total Stage 2 runs.

### Stage 2: Aux-Zero Continuations From New Boundary Handoffs

For each selected handoff checkpoint, run:

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
  --semantic-decoder-checkpoint <stage1_boundary_checkpoint> \
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

Use a new run root:

```text
runs/2026-05-08_phase4_boundary_closure
```

Recommended implementation:

- Either extend `scripts/run_phase4_min_supervision_boundary.py` with a
  `closure` subcommand;
- or add a small one-off runner
  `scripts/run_phase4_boundary_closure.py`.

Prefer a runner over ad hoc shell commands so the selected handoffs and summary
are reproducible.

## Fast Gates

For every new Stage 2 run, report:

- run path;
- selected checkpoint path;
- effective seed and CLI `--seed`;
- source Stage 1 checkpoint path;
- source Stage 1 step;
- source Stage 1 operand exact;
- final normal exact;
- final injection-zero exact;
- final forced-random exact;
- final oracle-at-eval exact;
- final operand exact;
- final pair exact;
- final calculator-result accuracy;
- mean A/B entropy and confidence;
- `final_aux_operand_loss_weight`;
- `final_adaptive_interface_loss_weight`;
- `final_input_proj_anchor_weight`;
- freeze settings;
- trainable parameter groups.

## Diagnostics

Run full diagnostics only where they can change the phase conclusion:

- any newly retained checkpoint that becomes a lower boundary than the previous
  retained checkpoint for that seed;
- any newly failed checkpoint that becomes the nearest lower failed neighbor;
- the final selected Phase 4 boundary set if it differs from the previous task.

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

This task succeeds if it produces one of these closure outcomes:

### Clean Phase-Wrap Outcome

- Seed `2` gets a narrower fail/retain bracket than `0.320` failed vs `0.641`
  retained.
- Seed `5` gets a failed lower neighbor below the `0.230` retained handoff, or
  the next lower measured handoff also retains and the report says explicitly
  that seed `5` remains unusually permissive.
- All retained boundary claims are supported by operand/pair/calculator
  accuracy, private protocol diagnostics, and full-enum gaps.

If this happens, recommend wrapping Phase 4.

### Confound Outcome

Any of these should block Phase 4 wrap:

- A newly retained checkpoint has high answer exact but weak operand/pair exact.
- Full-enum diagnostics show nonzero learned-minus-true or learned-minus-best
  gaps on a checkpoint that the fast gate would call retained.
- A lower handoff appears to recover by answer loss while private all-pair
  decoding remains substantially below the fast-gate sample.
- Seed `5` keeps retaining from very low handoffs, suggesting the boundary is
  not really about measured Stage 1 protocol quality for that seed.

If this happens, write a follow-up task focused on the confound. Do not wrap the
phase.

## Interpretation Rules

- Do not describe oracle-at-eval success as progress.
- Do not call answer exact alone retention.
- Call a checkpoint retained only when learned operand exact, pair exact,
  calculator-result accuracy, private all-pair decoding, and full-enum gaps
  agree.
- Treat seed `5` below-boundary recovery as important, not as noise to average
  away.
- Keep this task scoped to boundary closure. No upstream unfreezing, no new
  estimators, no new objectives.

## Deliverables

- New closure run root and runner path.
- Table of selected handoffs and why each was selected.
- Fast-gate metrics for every new Stage 2 run.
- Full diagnostics for boundary-changing retained/failure checkpoints.
- Updated final Phase 4 boundary table.
- Explicit recommendation: `wrap Phase 4` or `do one more confound task`.
- Fact-sheet and work-history updates.
- Move this task to `aiAgentProjectTasks/completed/phase4/` when complete.
- Commit and push.
