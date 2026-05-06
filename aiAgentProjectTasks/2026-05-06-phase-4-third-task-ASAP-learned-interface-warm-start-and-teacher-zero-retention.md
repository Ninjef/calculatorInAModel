# Phase 4 Third Task: ASAP Learned Interface Warm Start and Teacher-Zero Retention

## Claim

The only research question that matters now is whether the upstream/model-side
calculator interface can learn to provide useful calculator inputs, then retain
that protocol when direct teacher signals are removed.

Primary claim:

```text
Starting from the validated sum_left_operand semantic decoder, direct operand
supervision should teach calculator_hook.input_proj to emit true operands. The
next question is whether answer loss can preserve that learned calculator-query
protocol when aux_operand_loss_weight is exactly 0.0.
```

This task is not allowed to rediscover oracle downstream success. Use the
existing oracle semantic decoder as infrastructure.

## Starting Point

Use this Stage 0B semantic decoder checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

It has already passed the wiring gate:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- eval exact `1.000`
- injection-zero exact `0.0`
- forced-random exact `0.0273`
- oracle-at-eval exact `1.0`
- trace operand exact `1.0`
- calculator-result accuracy `1.0`

Do not rerun oracle-only controls unless this checkpoint is missing or the
calculator-output/answer-decoder wiring changed. If rebuilding the artifact is
unavoidable, label it as artifact reconstruction, not progress.

## Fastest Useful Plan

Run one seed first. Only expand to multiple seeds after the one-seed pipeline has
a nontrivial learned-interface result.

### Stage 1: Aux-Only Interface Warm Start

Goal: teach `calculator_hook.input_proj` to read the true operands from the
fixed prompt operand positions.

Training constraints:

- load the selected Stage 0B checkpoint;
- freeze semantic decoder and upstream encoder;
- train only `calculator_hook.input_proj`;
- use `answer_loss_weight=0.0`;
- use `aux_operand_loss_weight=1.0`;
- use `adaptive_interface_loss_weight=0.0`;
- keep `calculator_output_format=sum_left_operand`;
- do not run expensive diagnostics unless the fast gate passes.

Suggested command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 500 \
  --batch-size 64 \
  --eval-samples 512 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --calculator-read-position operands \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-estimator adaptive_interface \
  --semantic-decoder-checkpoint /Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt \
  --freeze-semantic-decoder \
  --freeze-upstream-encoder \
  --answer-loss-weight 0.0 \
  --adaptive-interface-loss-weight 0.0 \
  --aux-operand-loss-weight 1.0 \
  --aux-operand-loss-decay-steps 0 \
  --input-proj-lr 0.003 \
  --upstream-lr 0.003 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --seed 1 \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

Stage 1 fast gate:

- operand exact preferably `>= 0.95`;
- calculator-result accuracy preferably `>= 0.95`;
- normal exact should become high if the interface and semantic decoder agree;
- injection-zero and forced-random should remain near chance;
- trainable parameter groups must be limited to `calculator_hook.input_proj`.

If Stage 1 fails, stop and fix the warm-start mechanics before trying
retention.

### Stage 2: Teacher-Zero Retention

Goal: test whether the learned calculator-query protocol survives with direct
operand supervision exactly removed.

Start from the best Stage 1 checkpoint or final weights. Use the same frozen
semantic decoder/upstream setup, but set direct teacher loss to exactly zero.

Training constraints:

- load the selected Stage 1 handoff checkpoint;
- freeze semantic decoder and upstream encoder;
- train only `calculator_hook.input_proj`;
- set `aux_operand_loss_weight=0.0`;
- set `adaptive_interface_loss_weight=0.0`;
- use `answer_loss_weight=1.0`;
- select only checkpoints where aux is exactly `0.0`.

Suggested command template:

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
  --calculator-read-position operands \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-estimator adaptive_interface \
  --semantic-decoder-checkpoint <stage1-selected-or-final-weights.pt> \
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
  --seed 1 \
  --snapshot-every 50 \
  --checkpoint-every 50 \
  --snapshot-samples 256 \
  --log-every 50
```

Stage 2 fast gate:

- `final_aux_operand_loss_weight == 0.0`;
- normal exact remains meaningfully above injection-zero and forced-random;
- operand exact remains above `0.35` as a minimum useful signal, preferably much
  higher;
- calculator-result accuracy remains above `0.40` as a minimum useful signal;
- oracle-at-eval remains high, confirming the frozen semantic decoder is still
  usable;
- trainable parameter groups remain limited to `calculator_hook.input_proj`.

If Stage 2 immediately collapses, try one short bridge run before broad sweeps:

```text
aux_operand_loss_weight=0.1
aux_operand_loss_decay_steps=200
aux_operand_loss_floor=0.0
answer_loss_weight=1.0
```

Then select only snapshots after the aux weight has reached exactly `0.0`.

## Diagnostics Only After Fast Gates

Do not run full diagnostics just because a run exists. Run expensive diagnostics
only on:

- the Stage 1 handoff if it clears the warm-start gate;
- selected Stage 2 aux-zero checkpoints with nontrivial learned-interface
  retention;
- one failed Stage 2 checkpoint only if needed to understand collapse.

For selected checkpoints, run:

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

PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_private_protocol.py \
  --checkpoint <selected-weights.pt> \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --output-dir <run-dir>/private_protocol_diagnostics

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

## Deliverables

- Stage 1 run path and selected handoff checkpoint.
- Stage 1 fast-gate metrics, especially learned operand exact and
  calculator-result accuracy.
- Stage 2 run path and selected aux-zero checkpoint, if any.
- Confirmation that selected Stage 2 checkpoint has aux exactly `0.0`.
- Fast-gate metrics comparing normal, injection-zero, forced-random, and
  oracle-at-eval.
- Private protocol and full-enum diagnostics only for selected meaningful
  checkpoints.
- Fact-sheet and work-history updates.
- Commit and push.

## Go / No-Go

Go to seed replication only if a Stage 2 aux-zero checkpoint shows nontrivial
learned-interface retention.

No-go if Stage 1 cannot learn the operand protocol under direct supervision.

Do not claim progress from oracle-only metrics. Progress means the learned
interface emits useful calculator inputs.
