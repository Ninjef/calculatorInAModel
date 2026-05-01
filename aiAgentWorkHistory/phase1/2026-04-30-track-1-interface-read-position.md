# 2026-04-30 - Track 1 interface read position

Task: implement `calculator_read_position=operands` and run the first `operand_max=19` probe-guided calculator-interface pass.

## Implementation

- Added `GPTConfig.calculator_read_position`, defaulting to `eq`.
- Preserved the existing `eq` interface behavior.
- Added `operands` mode, where learned A logits are read from the final A digit residual and learned B logits are read from the final B digit residual, while result injection remains localized to `=`.
- Threaded the setting through training, diagnostics, saved configs/checkpoints, and the non-bottleneck experiment runner.
- Added trace metadata for read-position mode and effective A/B/`=` indices.
- Updated auxiliary operand loss to supervise the same effective read sites as the active calculator interface.

## Validation

```bash
PYTHONPATH=. pytest -q
python3 scripts/run_non_bottleneck_protocol_experiments.py --track ladder --operand-max 19 --calculator-read-position operands --dry-run
```

Result: `33 passed`.

## First rung: operand_max=19

All runs used:

```text
digits=2
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
operand_max=19
calculator_operand_vocab_size=20
seed argument=0
run seed=2
```

| Run | Exact match | Notes | Path |
| --- | ---: | --- | --- |
| Model A | `0.883` | Raw transformer baseline | `runs/2026-04-30_121156_701135_model-a-op0-19/model-a-2digit-seed2` |
| Model B | `0.977` | Hook-off control is healthy | `runs/2026-04-30_121243_326952_model-b-op0-19/model-b-2digit-seed2` |
| Model C learned | `0.902` | Does not beat B | `runs/2026-04-30_121327_706589_model-c-op0-19/model-c-2digit-seed2` |
| Model C oracle | `0.998` | Oracle path works | `runs/2026-04-30_121510_503634_model-c-oracle-op0-19/model-c-2digit-seed2` |

Learned Model C metrics:

- Training metrics diagnostic sample: operand exact match `0.000`, calculator result accuracy `0.031`.
- Counterfactuals: injection-zero `0.844`, forced-zero `0.898`, forced-random `0.898`, oracle-at-eval `0.930`.
- Probe diagnostic sample: exact match `0.879`, operand exact match `0.000`, calculator result accuracy `0.035`.
- Layer 1 probes:
  - A token: A `0.854`, B `0.068`.
  - B token: A `0.126`, B `0.621`.
  - `=` token: A `0.117`, B `0.097`.

Oracle Model C metrics:

- Training metrics diagnostic sample: operand exact match `1.000`, calculator result accuracy `1.000`.
- Counterfactuals: injection-zero `0.047`, forced-zero `0.008`, forced-random `0.031`, oracle-at-eval `1.000`.
- Layer 1 probes:
  - A token: A `0.796`, B `0.058`.
  - B token: A `0.087`, B `0.631`.
  - `=` token: A `0.087`, B `0.097`.

## Interpretation

This is a useful negative first rung. The operand-read interface is implemented and the oracle path remains strong and calculator-dependent. Probes confirm operand information is available at the operand read sites. But learned Model C still fails to learn the true operand protocol from answer loss alone and does not show a credible calculator-use signature relative to Model B.

Per the track guardrail, do not proceed to `operand_max=49`/`99` or 3-seed repeats from this result alone.
