# 2026-04-27 — Training-ready data batches

Task: Follow-up after steps 1 and 2. Make the synthetic arithmetic generator ready to feed a next-token transformer training loop.

## What I did

**Data module.**
- Added `<pad>` to the vocabulary and exposed `PAD_ID`.
- Added `ArithmeticBatch`, a small dataclass containing `x`, `y`, and `loss_mask` tensors.
- Added fixed-width operand generation by default, so examples like `93+09=102<eos>` have stable input structure for a given digit count.
- Kept natural-width generation available with `fixed_width=False`.
- Added `max_sequence_length(num_digits)` for predictable sequence sizing.
- Added `pad_sequence(...)`.
- Added `make_batch(...)`, which:
  - generates samples,
  - pads them,
  - shifts tokens into next-token `x` and `y`,
  - shifts the answer-only `loss_mask`,
  - returns PyTorch tensors on an optional target device.

**Masking fix.**
- Caught and fixed a subtle padding issue: loss masks must pad with `0`, not `PAD_ID`, otherwise padded target positions would become truthy after conversion to `torch.bool`.

**Sample script.**
- Updated `scripts/sample_data.py` to print a small training batch shape summary after the sample examples.

**Tests.**
- Added `tests/test_data.py` covering:
  - tokenizer/detokenizer round-trip for `<eos>` and `<pad>`,
  - answer-only loss mask behavior,
  - padding length validation,
  - fixed-width batch shapes and dtypes,
  - padded target positions staying masked out,
  - natural-width batches padding to the longest sample.
- Added `pytest.ini` to disable pytest cache writes, avoiding sandbox-related cache warnings.
- Added `pytest` to `requirements.txt`.

**Docs.**
- Updated `README.md` with the `python -m pytest` command.

## Verification

Ran:

```bash
python3 -m pytest
python3 scripts/sample_data.py
python3 scripts/check_env.py
```

Results:
- `python3 -m pytest`: 5 tests passed.
- `python3 scripts/sample_data.py`: printed seeded examples and a `(4, 9)` training batch for 2-digit addition.
- `python3 scripts/check_env.py`: torch import worked, but MPS still reported `mps available: False` in this environment, with `mps built: True`.

## Files changed

```
README.md
requirements.txt
pytest.ini
scripts/sample_data.py
src/data.py
tests/test_data.py
```

## Next suggested step

Implement Step 3: the tiny decoder-only transformer and an overfit-one-batch smoke test using `make_batch(...)`.
