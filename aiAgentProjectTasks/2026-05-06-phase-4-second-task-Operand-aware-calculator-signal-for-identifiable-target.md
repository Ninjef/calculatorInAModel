# Phase 4 Second Task: Operand-Aware Calculator Signal for Identifiable Target

## Claim

The newly implemented `sum_left_operand` answer format is not yet an
identifiable calculator-protocol task under the current strict
`answer_decoder` bottleneck, because the decoder receives only the calculator
result class `a+b`, not the selected operand identity.

Primary claim for this task:

```text
If the strict answer decoder receives an operand-aware calculator signal
containing both the sum and selected left operand, then oracle operands should
support high exact match on SSSAA<eos>, while the old sum-only calculator signal
should fail as an intentional negative control.
```

This is a wiring/control task before learned-interface retention. Do not start
Stage 1/2 retention until this task passes.

## Why This Is the Next Task

Phase 4 now has `--answer-format sum_left_operand`, but the current model path
does not make the left operand available downstream:

- `CalculatorHook` computes predicted or oracle `a_pred` and `b_pred`.
- It then collapses them to a one-hot calculator result class.
- `calculator_hook.output_proj` projects only that result class.
- `TinyGPT._answer_bottleneck_logits` feeds only `calculator_injection` plus
  answer-position offsets into `answer_decoder`.

Therefore, for two examples with the same sum but different left operands, the
strict answer decoder gets the same semantic input. It can learn `SSS`, but it
cannot reliably learn `AA` from the calculator signal. An oracle semantic
decoder trained under this sum-only signal is expected to fail or plateau far
below the Phase 4 oracle gate.

The next task must repair the bottleneck signal before experiments can test
protocol retention honestly.

## Required Code Changes

Add a configurable calculator output signal while preserving existing behavior
as the default.

Suggested flag:

```text
--calculator-output-format sum | sum_left_operand
```

Default must be:

```text
sum
```

Implement:

```text
sum
  calculator signal = one_hot(a + b)

sum_left_operand
  calculator signal = concat(one_hot(a + b), one_hot(a))
```

Recommended model/config updates:

- Add `calculator_output_format` to `GPTConfig`.
- Validate choices in `CalculatorHook`.
- Make `calculator_hook.output_proj` input width depend on
  `calculator_output_format`.
- Preserve existing checkpoint compatibility for default `sum`.
- Thread the new flag through:
  - `scripts/overfit_one_batch.py`
  - `scripts/diagnose_calculator_protocol.py`
  - `scripts/run_action_loss_diagnostic.py` / legacy Track 4 implementation
  - `scripts/run_full_enum_action_loss_diagnostic.py`
  - `scripts/diagnose_private_protocol.py`
- Save the flag in configs/metrics/diagnostic summaries.

Implementation notes:

- For oracle operands, the left-operand one-hot should use oracle `a`.
- For learned operands, the left-operand one-hot should use learned `a_pred`.
- If a straight-through helper is added for `one_hot(a_pred)`, keep it small,
  tested, and default-preserving.
- Forced-result controls should be documented carefully: under
  `sum_left_operand`, forcing the result class changes the sum component but
  may leave the left-operand component intact. Injection-zero remains the clean
  full-signal ablation.

## Required Tests

Add focused tests before running experiments:

- Default `calculator_output_format=sum` is unchanged.
- `sum_left_operand` output projection input width is
  `calculator_result_vocab_size + calculator_operand_vocab_size`.
- In `answer_decoder` mode, two same-sum prompts with different left operands
  remain indistinguishable under `calculator_output_format=sum`.
- In `answer_decoder` mode with `calculator_output_format=sum_left_operand`,
  oracle operands can produce different answer-position logits for same-sum
  prompts with different left operands.
- Training/diagnostic configs serialize the new flag.
- Existing tests still pass:

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

## Stage 0A: Sum-Only Negative Control

Train the new answer format with oracle operands but the old sum-only
calculator signal:

```text
--answer-format sum_left_operand
--calculator-output-format sum
--oracle-train
```

Purpose:

- Demonstrate the current bottleneck is underidentified for `SSSAA<eos>`.
- Avoid mistaking a failed oracle run for an optimization failure.

Expected result:

- Exact match should be far below the Phase 4 oracle gate.
- The model may learn the sum prefix but should not reliably emit the correct
  left operand for ambiguous same-sum pairs.

Do not use this checkpoint for Stage 1.

## Stage 0B: Operand-Aware Oracle Semantic Decoder

Train the strict answer decoder with oracle operands and the operand-aware
calculator output signal:

```text
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 1000 \
  --batch-size 64 \
  --eval-samples 512 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --oracle-train \
  --calculator-read-position operands \
  --calculator-bottleneck-mode answer_decoder \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --seed 0 \
  --log-every 50
```

Use the existing strict Phase 2/3 base regime unless implementation details
force a small adjustment.

Required gates:

- Oracle train/eval exact preferably `>= 0.90`.
- Oracle-at-eval recovers high exact match.
- Injection-zero and forced-random are near chance.
- Trace operand exact and calculator-result accuracy are `1.0` under oracle
  operands.

If this fails, stop and fix wiring or decoder capacity before any learned
interface run.

## Diagnostics for Stage 0B

Run the canonical diagnostics on the selected oracle semantic decoder
checkpoint:

```text
python3 -m scripts.run_causal_calculator_protocol_diagnostics \
  --checkpoint <stage0b-final-weights.pt> \
  --samples 256 \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --oracle \
  --forced-result-sweep \
  --forced-result-batch-size 64 \
  --output-dir <stage0b-run-dir>/canonical_causal_diagnostics
```

Run full-enum only if Stage 0B passes the oracle gate or if needed to debug:

```text
python3 scripts/run_full_enum_action_loss_diagnostic.py \
  --checkpoint <stage0b-final-weights.pt> \
  --samples 128 \
  --batch-size 64 \
  --digits 2 \
  --operand-max 19 \
  --answer-format sum_left_operand \
  --calculator-output-format sum_left_operand \
  --temperature 1.0 \
  --chunk-size 64 \
  --output-root <stage0b-run-dir>/full_enum_action_loss
```

## Deliverables

- Code and tests for `calculator_output_format`.
- Stage 0A negative-control run path and metrics.
- Stage 0B oracle semantic decoder run path and metrics.
- Selected Stage 0B checkpoint path for the next Stage 1 task.
- Fact-sheet and work-history updates.
- Commit and push.

## Go / No-Go

Go to Stage 1 supervised interface warm start only if:

- Stage 0A confirms sum-only is insufficient or clearly weaker;
- Stage 0B reaches high oracle exact match;
- counterfactual controls show calculator dependence;
- the selected semantic decoder checkpoint is recorded.

No-go if Stage 0B cannot solve `SSSAA<eos>` with oracle operands.
