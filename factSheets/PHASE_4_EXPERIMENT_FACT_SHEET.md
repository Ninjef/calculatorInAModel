# Phase 4 Experiment Fact Sheet

## Direction

Phase 4 tests whether a calculator-query protocol can be taught and retained
when the downstream answer target makes operand identity useful.

The first implemented identifiable target keeps the prompt shape:

```text
AA+BB=
```

and adds an opt-in answer format:

```text
SSSAA<eos>
```

where `SSS` is the zero-padded sum and `AA` is the zero-padded left operand.

Example:

```text
07+12=01907<eos>
```

## 2026-05-06 Implementation Facts

- Added `--answer-format sum | sum_left_operand`.
- Default `sum` behavior is preserved.
- `sum_left_operand` is digit-only and needs no vocabulary changes.
- The model architecture is unchanged; only block/sequence length follows the answer format.
- Training and checkpoint diagnostics can now be run with `--answer-format sum_left_operand`.
- Calculator operand read positions remain anchored to the fixed-width prompt operands, independent of the longer answer suffix.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/data.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
66 passed
```

## 2026-05-06 Operand-Aware Calculator Signal

Claim tested:

```text
For answer_format=sum_left_operand, the strict answer decoder needs an
operand-aware calculator output signal; sum-only is an intentional negative
control.
```

Implementation facts:

- Added `--calculator-output-format sum | sum_left_operand`; default is `sum`.
- `sum` preserves the old calculator signal: `one_hot(a+b)`.
- `sum_left_operand` projects `concat(one_hot(a+b), one_hot(a))`.
- Oracle operands use oracle `a`; learned operands use learned `a_pred`, with a
  small straight-through one-hot helper for the independent learned A head.
- The `sum_left_operand` strict decoder path now includes a calculator-signal /
  answer-offset interaction. The old `sum` decoder path remains additive for
  compatibility.
- Training configs, metrics, and diagnostic summaries serialize
  `calculator_output_format`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
71 passed
```

Stage 0A negative control:

- Command used oracle operands with `--answer-format sum_left_operand` and
  `--calculator-output-format sum`.
- Run path:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_163931_767398_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2`
- Eval exact: `0.043` (`22/512`).
- Injection-zero exact: `0.0`; forced-random exact: `0.0`.
- Trace operand exact and calculator-result accuracy under oracle operands:
  `1.0`.

Stage 0B operand-aware oracle semantic decoder:

- Command used oracle operands with `--answer-format sum_left_operand` and
  `--calculator-output-format sum_left_operand`.
- Run path:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2`
- Selected checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- Eval exact: `1.000` (`512/512`), final loss `0.000224`.
- Canonical diagnostics (`samples=256`): normal exact `1.0`,
  injection-zero exact `0.0`, forced-zero exact `0.0039`,
  forced-random exact `0.0273`, oracle-at-eval exact `1.0`,
  operand exact `1.0`, calculator-result accuracy `1.0`.
- Canonical classification: `valid_oracle_calculator_use` /
  `calculator_required_bottleneck`.
- Full-enum diagnostic (`samples=128`): true-best fraction `1.0`, learned-best
  fraction `0.0078`, mean learned-minus-true gap `9.3793`. This is expected for
  the oracle semantic decoder because the learned interface head is still
  untrained/random.

Go recommendation:

- Go to Stage 1 supervised interface warm start using the selected Stage 0B
  checkpoint above.
