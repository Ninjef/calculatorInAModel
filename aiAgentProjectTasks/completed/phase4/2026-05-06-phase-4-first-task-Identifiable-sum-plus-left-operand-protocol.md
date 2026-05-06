# Phase 4 First Task: Identifiable Sum-plus-Left-Operand Protocol

## Claim

The current addition-only task is underidentified: many calculator operand pairs
can produce the same sum-like downstream signal. This task creates the smallest
digit-only identifiable extension and asks whether the calculator interface can
learn and retain a protocol when the answer requires operand identity.

Primary claim:

```text
If the answer target includes both the sum and the left operand, then a retained
aux-zero calculator protocol should become more true-operand-like than the best
Phase 2/3 addition-only checkpoints.
```

This is a protocol-teaching and retention task, not a pure answer-loss discovery
claim.

## Minimal Task Design

Keep the existing prompt format:

```text
AA+BB=
```

Use a fixed-width, digit-only answer target:

```text
SSSAA<eos>
```

where:

- `SSS` is the sum `a + b`, zero-padded to `num_digits + 1`.
- `AA` is the left operand `a`, zero-padded to `num_digits`.

Example:

```text
07+12=01907<eos>
```

Why this variant:

- It requires no new vocab tokens.
- It preserves the existing calculator prompt shape and operand-read positions.
- It makes true `a` identity directly useful under the `answer_decoder` bottleneck.
- It avoids adding a second operation before the first identifiability test.

## Required Code Changes

Add a configurable answer format while preserving all existing addition behavior
as the default.

Suggested flag:

```text
--answer-format sum | sum_left_operand
```

Default must be `sum`.

Implement `sum_left_operand` in the places that currently construct targets:

- `src/data.py`
- `scripts/overfit_one_batch.py`
- `scripts/diagnose_calculator_protocol.py`
- `scripts/run_action_loss_diagnostic.py`
- `scripts/run_full_enum_action_loss_diagnostic.py`
- `scripts/diagnose_private_protocol.py`, if needed for exact answer comparisons

Keep the model architecture unchanged except for block-size/sequence-length
calculation. The fixed-width target is longer than plain addition, so
`max_sequence_length` or the training script's configured block size must account
for the answer format.

## Required Tests

Add focused tests before running experiments:

- Tokenization still round-trips the new digit-only samples.
- `sum_left_operand` samples have the expected fixed-width string, for example
  `07+12=01907<eos>`.
- Loss masks still begin only after `=`.
- Batch shapes increase correctly for the longer answer.
- Existing default `sum` behavior is unchanged.
- Calculator read positions for operands remain unchanged.

Run:

```text
python3 -m pytest tests/test_data.py tests/test_model.py -q
```

## Experimental Setup

Use the existing strict Phase 2/3 base regime unless the implementation forces a
small adjustment:

```text
variant=model-c
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
calculator_bottleneck_mode=answer_decoder
freeze_semantic_decoder=true for retention runs
```

Because the answer format changes, do not reuse the old addition-only semantic
decoder as the primary semantic checkpoint. First train a new oracle semantic
decoder for `sum_left_operand`.

## Stage 0: Oracle Semantic Decoder Control

Train the strict `answer_decoder` bottleneck with oracle operands on the new
answer format.

Purpose:

- Prove the downstream answer decoder can emit `SSSAA<eos>` from correct
  calculator actions.
- Create the semantic decoder checkpoint for retention experiments.

Required gates:

- Oracle train/eval exact should be high, preferably `>= 0.90`.
- Injection-zero and forced-random should be near chance.
- Oracle-at-eval should recover high exact match.

If this fails, stop and fix the data/model wiring before trying learned
interfaces.

## Stage 1: Supervised Interface Warm Start

Train only the calculator input interface from direct true-operand supervision
against the new oracle semantic decoder.

Primary constraints:

```text
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=0.0 or low
aux_operand_loss_weight > 0.0
adaptive/action-loss weight=0.0 for the cleanest warm start
```

Purpose:

- Establish that the interface can read true operands from the existing read
  positions in the identifiable setting.
- Produce a handoff checkpoint for aux-zero retention.

Record:

- operand exact match;
- calculator-result accuracy;
- answer exact;
- injection-zero / forced-random / oracle-at-eval;
- final aux weight.

## Stage 2: Aux-Zero Retention

Start from the Stage 1 handoff and decay direct operand supervision to exactly
`0.0`, then continue with no direct supervision.

Primary constraints:

```text
final_aux_operand_loss_weight=0.0
final_input_proj_anchor_weight=0.0 unless explicitly testing anchors
freeze_semantic_decoder=true
freeze_upstream_encoder=true for the first retention ladder
trainable groups limited to calculator_hook.input_proj
```

Use dense checkpointing and select only among checkpoints where aux is exactly
`0.0`.

Suggested compact ladder:

```text
seeds: 1, 2, 3
snapshot_every: 50
checkpoint_every: 50
steps: 1000
input_proj_lr: 0.0003
```

Do not unfreeze upstream in this task unless the frozen-upstream aux-zero
retention result is already strong.

## Fast Gates

For every snapshot, record:

- normal exact match;
- injection-zero exact match;
- forced-random exact match;
- oracle-at-eval exact match;
- operand exact match;
- calculator-result accuracy;
- mean operand confidence/entropy;
- final/current aux weight;
- trainable parameter groups.

Run full diagnostics only on:

- the Stage 0 oracle control;
- the Stage 1 handoff;
- selected aux-zero Stage 2 checkpoints;
- one negative control if the result collapses.

## Full Diagnostics for Selected Checkpoints

For selected checkpoints, run:

```text
python3 -m scripts.run_causal_calculator_protocol_diagnostics ...
python3 scripts/run_full_enum_action_loss_diagnostic.py ...
python3 scripts/diagnose_private_protocol.py ...
```

All diagnostics must use `--answer-format sum_left_operand` once that flag
exists.

Report:

- canonical causal classification;
- injection-zero, forced-zero, forced-random, oracle-at-eval;
- private all-pair answer exact;
- operand exact match;
- calculator-result accuracy;
- group behavior for carry/no-carry, small/large, symmetric;
- full-enum learned, true, and best NLL;
- learned-minus-true and learned-minus-best gaps.

## Comparisons

Compare selected aux-zero checkpoints against:

- Best Phase 2 retained independent-head checkpoints:
  - private all-pair answer about `0.5300..0.5375`;
  - operand exact about `0.5500..0.5675`;
  - calculator-result accuracy about `0.5750..0.5775`;
  - learned-minus-true gaps around `1.9814..2.5646`.
- Best Phase 3 joint-pair checkpoints:
  - private pair exact around `0.10..0.14`;
  - private calculator-result accuracy around `0.15..0.18`;
  - pair logits near-uniform over roughly all `400` actions.

## Success Criteria

Strong positive:

- Stage 0 oracle exact `>= 0.90`.
- At least two of three Stage 2 seeds retain aux-zero operand exact above the
  best Phase 2 range.
- Calculator-result accuracy beats the best Phase 2 range.
- Injection-zero and forced-random remain near chance.
- Oracle-at-eval remains high.
- Learned-minus-true action-loss gap improves over the Phase 2 selected range.

Weak positive:

- Stage 2 aux-zero retention beats Phase 3 joint-pair results clearly but does
  not beat the best Phase 2 independent-head retention.
- The learned protocol is more true-operand-like than addition-only Phase 3.

Negative:

- Oracle control works, but aux-zero retention collapses.
- The identifiable target improves answer accuracy without improving operand
  exact or calculator-result accuracy.
- Upstream or answer leakage explains the result under counterfactuals.

## Stop Conditions

Stop early if:

- Stage 0 oracle control fails.
- The new answer format breaks existing default addition tests.
- Aux-zero snapshots stay below `0.20` operand exact across all seeds.
- Injection-zero or forced-random approaches normal exact, indicating bypass.

## Deliverables

End the task with:

- Code changes and tests run.
- Run paths for Stage 0, Stage 1, and Stage 2.
- Selected aux-zero checkpoints and why they were selected.
- Fast-gate table.
- Full diagnostics for selected checkpoints only.
- Direct comparison to Phase 2 and Phase 3 baselines.
- Recommendation: continue identifiable-task work, adjust protocol teaching, or stop this branch.
