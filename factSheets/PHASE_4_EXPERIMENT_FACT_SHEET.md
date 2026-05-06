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
