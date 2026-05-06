# 2026-05-06 - Operand-aware calculator output signal

Task: make `answer_format=sum_left_operand` identifiable under the strict
`answer_decoder` bottleneck by adding an operand-aware calculator output signal.

## Implementation

- Added `GPTConfig.calculator_output_format` with choices `sum` and
  `sum_left_operand`; default remains `sum`.
- Preserved the old calculator signal as `one_hot(a+b)`.
- Added `sum_left_operand` calculator signal as
  `concat(one_hot(a+b), one_hot(a))`.
- For oracle operands, the left-operand component uses oracle `a`.
- For learned operands, the left-operand component uses learned `a_pred`; the
  independent learned A head gets a small straight-through one-hot helper.
- Made `calculator_hook.output_proj` input width depend on the calculator output
  format.
- Threaded `--calculator-output-format` through the training and diagnostic
  CLIs and serialized it in configs, metrics, and diagnostic summaries.
- Added a signal/offset interaction for the strict decoder when
  `calculator_output_format=sum_left_operand`. This fixed the oracle decoder
  capacity issue while leaving the default `sum` path on the previous additive
  decoder behavior.

## Tests

Added focused tests for:

- default `calculator_output_format=sum` projection width;
- `sum_left_operand` projection width;
- invalid output-format validation;
- same-sum oracle prompts staying indistinguishable with sum-only signal;
- same-sum oracle prompts becoming distinguishable with operand-aware signal;
- training config and metrics serialization of `calculator_output_format`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
71 passed
```

## Stage 0A: Sum-only negative control

Command shape:

```bash
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 1000 \
  --batch-size 64 \
  --eval-samples 512 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --answer-format sum_left_operand \
  --calculator-output-format sum \
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

Run path:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_163931_767398_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2
```

Metrics:

- Eval exact: `0.043` (`22/512`).
- Diagnostic exact: `0.0859`.
- Injection-zero exact: `0.0`.
- Forced-random exact: `0.0`.
- Oracle-at-eval exact: `0.0469`.
- Operand exact and calculator-result accuracy under oracle operands: `1.0`.

Conclusion: sum-only remains far below the oracle gate on `SSSAA<eos>`, as
expected.

## Stage 0B: Operand-aware oracle semantic decoder

Command shape:

```bash
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

Run path:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Selected checkpoint for Stage 1:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Metrics:

- Eval exact: `1.000` (`512/512`).
- Final loss: `0.000224`.
- Trace operand exact: `1.0`.
- Calculator-result accuracy: `1.0`.

Canonical diagnostics:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/canonical_causal_diagnostics
```

- Samples: `256`.
- Normal exact: `1.0`.
- Injection-zero exact: `0.0`.
- Forced-zero exact: `0.0039`.
- Forced-random exact: `0.0273`.
- Oracle-at-eval exact: `1.0`.
- Classification: `valid_oracle_calculator_use` /
  `calculator_required_bottleneck`.

Full-enum diagnostics:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/full_enum_action_loss/model-c-2digit-seed2
```

- Samples: `128`.
- True-best fraction: `1.0`.
- Learned-best fraction: `0.0078`.
- Mean learned-minus-true gap: `9.3793`.
- This is expected for an oracle semantic decoder: the learned interface head is
  still untrained/random.

## Recommendation

Go to Stage 1 supervised interface warm start from the selected Stage 0B
checkpoint. Do not use the Stage 0A checkpoint except as the negative control.
