# 2026-04-27 — Steps 1 & 2: Environment + synthetic data generator

Task: `aiAgentProjectTasks/2026-04-27-1134-Set-Up-and-training-data.md` — implement only steps 1 and 2 of the calculator-in-the-brain research plan.

## What I did

**Planning.** Wrote a plan at `~/.claude/plans/sorry-please-create-goofy-gosling.md` covering both steps. Asked the user two clarifying questions before exiting plan mode:
- Python env → chose isolated `.venv/` at repo root with pinned `requirements.txt`.
- Sample format → chose full token sequence in natural digit order (caller shifts by one for input/target), with a loss-mask helper.

**Step 1 — Environment.**
- Created `.venv/` at repo root.
- Added `requirements.txt` with `torch`. Added `numpy` after first run because torch printed a noisy "Failed to initialize NumPy" warning on import.
- Added `.gitignore` for `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `runs/`.
- Wrote `scripts/check_env.py` — prints python/platform/torch versions, `mps available`, `mps built`, and runs a tiny `(x @ x.T).sum()` smoke op on the MPS device.
- Verified: torch 2.11.0, `mps available: True`, smoke op produced a scalar.

**Step 2 — Synthetic data generator.**
- Wrote `src/data.py` with no torch dependency (pure Python at the time):
  - 13-token vocab: digits `0`–`9`, `+`, `=`, `<eos>`.
  - `TOKEN_TO_ID`, `ID_TO_TOKEN`, `VOCAB_SIZE`, `PLUS_ID`, `EQ_ID`, `EOS_ID`.
  - `tokenize(s)` — single-char split, special-cased multi-char `<eos>`.
  - `detokenize(ids)` — inverse.
  - `generate_sample(num_digits, rng)` — operands uniform on `[0, 10**num_digits - 1]`, returns ids of `f"{a}+{b}={a+b}<eos>"`.
  - `generate_batch(batch_size, num_digits, rng)` — list of unpadded sequences.
  - `make_loss_mask(ids)` — 0s up through `=`, 1s after.
- Wrote `scripts/sample_data.py` to print 20 samples (7 × 1-digit, 7 × 2-digit, 6 × 3-digit) with seeded RNG, showing text/ids/mask.
- Verified: spot-checked sums (`6+6=12`, `36+17=53`, `483+573=1056`, `625+655=1280` etc.) and confirmed mask alignment.

**Hygiene.** Wrote a 5-line `README.md` covering venv setup and the two run commands.

## Files created

```
.gitignore
README.md
requirements.txt
src/__init__.py
src/data.py
scripts/check_env.py
scripts/sample_data.py
```

## Out of scope (deferred to later steps per the task doc)

Transformer model (Step 3), training loop (Step 4), run-directory hygiene (Step 5), calculator hook + STE (Steps 6–7).
