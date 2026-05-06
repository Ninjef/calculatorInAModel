# 2026-05-06 - Sum-plus-left-operand identifiable answer format

Task: implement the smallest digit-only identifiable extension for Phase 4,
preserving existing addition behavior by default.

## Implementation

- Added shared answer-format helpers in `src/data.py`.
- Added `sum_left_operand` target construction: `SSSAA<eos>`.
- Updated fixed-width sequence length and model block-size calculation to account for the longer target.
- Threaded `--answer-format` through:
  - `scripts/overfit_one_batch.py`
  - `scripts/diagnose_calculator_protocol.py`
  - `scripts/run_action_loss_diagnostic.py` via the legacy Track 4 implementation
  - `scripts/run_full_enum_action_loss_diagnostic.py`
  - `scripts/diagnose_private_protocol.py`
  - `scripts/run_action_loss_candidate_diagnostic.py`
- Preserved all existing `sum` defaults.

## Tests

Added focused tests for:

- round-tripping `07+12=01907<eos>`;
- exact `sum_left_operand` target formatting;
- answer loss masks starting only after `=`;
- longer batch shapes under the new format;
- unchanged default `sum` behavior;
- unchanged operand read positions with a longer answer suffix.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/data.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
66 passed
```

## Notes

This completed the implementation and focused test gate for the first Phase 4
task. It did not run Stage 0/1/2 experiments; those are now unblocked and
should use `--answer-format sum_left_operand`.
