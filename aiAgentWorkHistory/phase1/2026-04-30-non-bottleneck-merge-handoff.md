# 2026-04-30 - Merge handoff for non-bottleneck protocol experiments

Branch:

```text
experiment/non-bottleneck-protocol-experiments
```

Worktree used:

```text
/Users/jarnold/workspace/jeffarnoldlabs/sandboxes/calculatorInAModel
```

This branch intentionally does not implement the replacement/bottleneck experiment. It implements infrastructure and a first diagnostic pass for the non-bottleneck protocol-learning plan.

## What changed

Code:

- `scripts/overfit_one_batch.py`
  - Added `--oracle-warmup-steps`.
  - Added `--aux-operand-loss-floor`.
  - Added periodic Model C snapshots with `--snapshot-every` and `--snapshot-samples`.
  - Added final Model C counterfactual metrics to `metrics.json`:
    - injection scale zero;
    - oracle operands at eval;
    - forced zero calculator result;
    - forced random calculator result.
- `scripts/run_non_bottleneck_protocol_experiments.py`
  - New orchestration helper for the ladder, aux, warmup, private-code trajectory, and probe tracks.
- `tests/test_model.py`
  - Added coverage for aux-floor scheduling, oracle warmup config serialization, snapshot output, and counterfactual metrics.

Docs:

- `aiAgentWorkHistory/2026-04-30-non-bottleneck-protocol-experiments.md`
  - Records the experiment results and interpretation.
- `aiAgentWorkHistory/2026-04-30-non-bottleneck-merge-handoff.md`
  - This handoff.

Run artifacts were written under `runs/`, but this repo has historically treated large run artifacts as local outputs. Do not add them to git unless the project owner explicitly wants that.

## Verification already run

```bash
python3 -m pytest
```

Result:

```text
31 passed
```

Also run:

```bash
python3 scripts/run_non_bottleneck_protocol_experiments.py --track aux --dry-run
python3 scripts/run_non_bottleneck_protocol_experiments.py --track warmup --dry-run
```

Both dry-runs printed the expected commands.

One attempted check failed because the environment could not write `scripts/__pycache__`:

```bash
python3 -m py_compile scripts/run_non_bottleneck_protocol_experiments.py scripts/overfit_one_batch.py
```

That failure was filesystem-permission related, not a Python syntax failure. The full pytest run exercised both scripts successfully.

## Experiment results to understand before merging

Initial `2L/1H/16d/mlp1`, 2-digit, `operand_max=19` runs:

| Variant | Exact match | Run |
| --- | ---: | --- |
| Model A | `0.949` | `runs/2026-04-30_112240_823061_model-a-op0-19/model-a-2digit-seed2` |
| Model B | `0.627` | `runs/2026-04-30_112323_437684_model-b-op0-19/model-b-2digit-seed2` |
| Model C learned STE | `0.557` | `runs/2026-04-30_112404_779006_model-c-op0-19/model-c-2digit-seed2` |
| Model C oracle | `1.000` | `runs/2026-04-30_112536_945624_model-c-oracle-op0-19/model-c-2digit-seed2` |
| Model C + aux `0.01` | `0.777` | `runs/2026-04-30_112653_137515_model-c-op0-19-aux0.01/model-c-2digit-seed2` |
| Model C oracle warmup `100` + aux `0.01` | `0.049` | `runs/2026-04-30_112837_118416_model-c-oraclewarm100-op0-19-aux0.01/model-c-2digit-seed2` |

Interpretation:

- `operand_max=19` is not a clean headline capacity regime because Model A is already high.
- Model B is much lower than Model A, so the hook-off control is suspicious in this tiny setting.
- Oracle Model C works perfectly, confirming the output/injection side is usable.
- Aux `0.01` improves answer accuracy but does not produce convincing calculator dependence:
  - normal exact match `0.777`;
  - injection-zero exact match `0.742`;
  - forced-zero exact match `0.766`;
  - oracle-at-eval exact match `0.680`;
  - true operand exact match `0.000`.
- Oracle warmup with learned handoff failed badly. After the handoff, answer loss rose and aux operand loss exploded.

Probe result:

- A/B/oracle-C all show operand information at operand-token positions.
- All are weak at the current calculator read position, the `=` token.
- This suggests the next productive experiment is not another loss schedule on the same readout, but a calculator interface that reads operands from operand-token positions and still injects the result at `=`.

## How to merge if nothing else landed first

1. Inspect the diff:

```bash
git diff main...experiment/non-bottleneck-protocol-experiments -- scripts/overfit_one_batch.py scripts/run_non_bottleneck_protocol_experiments.py tests/test_model.py aiAgentWorkHistory/2026-04-30-non-bottleneck-protocol-experiments.md aiAgentWorkHistory/2026-04-30-non-bottleneck-merge-handoff.md
```

2. Re-run tests:

```bash
python3 -m pytest
```

3. Merge normally:

```bash
git switch main
git merge --no-ff experiment/non-bottleneck-protocol-experiments
```

4. Do not add `runs/` artifacts unless explicitly requested.

## How to merge if another experiment landed first

Most likely conflict files:

- `scripts/overfit_one_batch.py`
- `tests/test_model.py`
- possibly `scripts/diagnose_calculator_protocol.py` if the other experiment extended diagnostics, though this branch did not edit that file.

Resolution guidance:

- Keep all existing defaults compatible. The new flags must default to disabled:
  - `--oracle-warmup-steps 0`;
  - `--aux-operand-loss-floor 0.0`;
  - `--snapshot-every 0`;
  - `--snapshot-samples 64`.
- Preserve existing `--oracle-train` semantics. `--oracle-train` means oracle operands throughout training and final eval. `--oracle-warmup-steps` means oracle operands only for the first K training steps, with normal learned-operand final eval.
- Preserve the final `metrics.json` shape for existing keys. The new `counterfactuals` object is additive and only present for Model C runs.
- If another branch added calculator injection modes or read-position modes, thread those through `TrainConfig.model` and the runner script rather than creating separate config systems.
- If another branch edited aux loss behavior, keep `auxiliary_operand_weight(...)` as the single schedule helper or merge both schedules into one helper with explicit tests.
- Keep `scripts/run_non_bottleneck_protocol_experiments.py` as a thin command launcher. It should not grow training logic; training behavior belongs in `overfit_one_batch.py`.

After conflict resolution, run:

```bash
python3 -m pytest
python3 scripts/run_non_bottleneck_protocol_experiments.py --track ladder --operand-max 19 --dry-run
python3 scripts/run_non_bottleneck_protocol_experiments.py --track aux --dry-run
python3 scripts/run_non_bottleneck_protocol_experiments.py --track warmup --dry-run
```

## What not to infer from this branch

- Do not treat the `0..19` results as evidence that the calculator idea failed.
- Do not treat the aux `0.01` improvement as calculator use; counterfactuals say it is mostly bypass/ordinary answer learning.
- Do not spend a large run budget on more seeds for the same `=` readout until the read-position issue is addressed.

## Recommended next branch

Implement a narrow read-position interface:

```text
calculator_read_position=eq        # default, current behavior
calculator_read_position=operands  # read A logits from final A digit, B logits from final B digit
```

Keep calculator result injection at `=`.

Then re-run the `operand_max=19` and `49` ladder with A/B/C/oracle controls. This directly tests the probe-guided hypothesis from this branch.
