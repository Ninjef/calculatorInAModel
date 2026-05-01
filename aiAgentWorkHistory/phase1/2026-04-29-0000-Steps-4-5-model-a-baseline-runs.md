# 2026-04-29 — Steps 4-5: Model A baseline runs + run hygiene

Task: Implement Steps 4 and 5 from the research plan — train the raw transformer baseline on 1-, 2-, and 3-digit addition, and establish a clean saved-run structure.

## What I did

**Baseline trainer.**
- Reworked `scripts/overfit_one_batch.py` from a fixed-batch overfit smoke test into a Model A baseline runner.
- Added CLI options for digit counts, steps, batch size, eval sample count, learning rate, seed, logging cadence, gradient clipping, weight decay, and run root.
- Each digit count is trained as a separate run with a `GPTConfig` block size matched to the task length.
- Evaluation uses greedy generation from prompts like `03+04=` and computes exact-match accuracy against the generated answer plus `<eos>`.

**Run hygiene.**
- Added run directories under `runs/YYYY-MM-DD_HHMMSS_model-a-baseline/`.
- Each per-digit run saves:
  - `config.json`
  - `final_weights.pt`
  - `training_curve.csv`
  - `metrics.json`
- The parent run directory also saves `summary_metrics.json`.

**Manual probing helper.**
- Added `scripts/try_model.py` so trained checkpoints can be tested interactively or from the command line.
- Example:

```bash
python3 scripts/try_model.py runs/2026-04-28_185900_model-a-baseline/model-a-1digit-seed1 3+4= 8+7=
```

## Baseline results

Ran:

```bash
python3 scripts/overfit_one_batch.py
```

Saved run:

```text
runs/2026-04-28_185900_model-a-baseline
```

Results:

```text
1-digit: 228/256 exact match = 0.890625, final loss = 0.4046
2-digit:  22/256 exact match = 0.085938, final loss = 0.7580
3-digit:   1/256 exact match = 0.003906, final loss = 1.0929
```

This gives a useful control condition: the raw transformer learns much of 1-digit addition, struggles badly on 2-digit addition, and essentially fails on 3-digit addition under the default training budget.

## Verification

Ran:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider
python3 scripts/try_model.py runs/2026-04-28_185900_model-a-baseline/model-a-1digit-seed1 3+4= 8+7=
python3 scripts/try_model.py runs/2026-04-28_185900_model-a-baseline/model-a-2digit-seed2 03+04= 43+29=
```

Results:
- `pytest`: 8 tests passed.
- `try_model.py` successfully loaded saved checkpoints and generated completions.

## Files changed

```text
scripts/overfit_one_batch.py
scripts/try_model.py
runs/2026-04-28_185900_model-a-baseline/...
```

## Next suggested step

Step 6: build Model B by adding the calculator hook with the calculator effectively off, then confirm it matches Model A within noise before turning the calculator on.
