# 2026-04-30 - Non-bottleneck calculator protocol experiments

Task: Implement and start the non-bottleneck experiment plan from `aiAgentProjectTasks/2026-04-29-1320-Next-calculator-protocol-experiments.md`, explicitly avoiding the replacement bottleneck follow-up.

## Code changes

- Added oracle-to-learned warmup support to `scripts/overfit_one_batch.py` via `--oracle-warmup-steps`.
- Added auxiliary operand loss floors via `--aux-operand-loss-floor`, so schedules like `0.1` decayed to a `0.01` floor are expressible.
- Added periodic Model C diagnostic snapshots via `--snapshot-every` and `--snapshot-samples`.
- Added final Model C counterfactual metrics to `metrics.json`:
  - injection scale zero;
  - oracle operands at eval;
  - forced zero result;
  - forced random result.
- Added `scripts/run_non_bottleneck_protocol_experiments.py` to launch the ladder, aux, warmup, private-code trajectory, and probe tracks reproducibly.
- Added tests for aux floor scheduling, oracle warmup config serialization, snapshot output, and final counterfactual metrics.

Verification:

```bash
python3 -m pytest
```

Result: `31 passed`.

## Initial 2-digit `0..19` ladder rung

All runs used:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
steps=1000
eval_samples=512
seed argument=0
```

| Variant | Exact match | Final loss | Run |
| --- | ---: | ---: | --- |
| Model A | `0.949` | `0.0763` | `runs/2026-04-30_112240_823061_model-a-op0-19/model-a-2digit-seed2` |
| Model B | `0.627` | `0.2487` | `runs/2026-04-30_112323_437684_model-b-op0-19/model-b-2digit-seed2` |
| Model C learned STE | `0.557` | `0.2703` | `runs/2026-04-30_112404_779006_model-c-op0-19/model-c-2digit-seed2` |
| Model C oracle | `1.000` | `0.0000` | `runs/2026-04-30_112536_945624_model-c-oracle-op0-19/model-c-2digit-seed2` |
| Model C learned STE + aux `0.01` | `0.777` | `0.1852` | `runs/2026-04-30_112653_137515_model-c-op0-19-aux0.01/model-c-2digit-seed2` |
| Model C oracle warmup `100` + aux `0.01` | `0.049` | `4.4125` | `runs/2026-04-30_112837_118416_model-c-oraclewarm100-op0-19-aux0.01/model-c-2digit-seed2` |

Interpretation:

- `0..19` is not a clean capacity regime because Model A is already high at `0.949`.
- Model B being far below Model A means this narrow hook-off architecture is not a fully benign control here.
- Oracle still works perfectly, so the calculator output path is usable.
- Aux `0.01` improves answer accuracy, but not in a calculator-dependent way:
  - normal exact match: `0.777`;
  - injection-zero exact match: `0.742`;
  - forced-zero exact match: `0.766`;
  - oracle-at-eval exact match: `0.680`;
  - true operand exact match: `0.000`.
- Oracle warmup with immediate handoff is unstable. After oracle operands turn off, answer loss jumps and aux operand loss explodes, ending at `0.049` exact match.

## Probe-guided finding

Probe runs:

```text
runs/2026-04-30_112240_823061_model-a-op0-19/model-a-2digit-seed2/diagnostics
runs/2026-04-30_112323_437684_model-b-op0-19/model-b-2digit-seed2/diagnostics
runs/2026-04-30_112536_945624_model-c-oracle-op0-19/model-c-2digit-seed2/diagnostics
```

Key probe pattern on 1024 samples, 400 probe steps:

| Checkpoint | Layer/position | Operand A probe | Operand B probe |
| --- | --- | ---: | ---: |
| Model A | layer 1, A token | `1.000` | `0.049` |
| Model A | layer 1, B token | `0.137` | `0.693` |
| Model A | layer 1, `=` token | `0.122` | `0.107` |
| Model B | layer 1, A token | `0.961` | `0.054` |
| Model B | layer 1, B token | `0.112` | `0.673` |
| Model B | layer 1, `=` token | `0.122` | `0.132` |
| Oracle C | layer 1, A token | `0.810` | `0.049` |
| Oracle C | layer 1, B token | `0.098` | `0.688` |
| Oracle C | layer 1, `=` token | `0.088` | `0.141` |

Interpretation:

- Operand information is linearly accessible at the operand token positions.
- Operand information is weak at the current calculator read position, `=`.
- The learned STE failure is therefore likely not just class count or loss shaping; the interface asks a single `=` residual vector to expose both operands when the model naturally stores them elsewhere.

## Recommended next move

Do not spend the next block on more seeds for the same `=`-read interface.

Implement a narrow non-bottleneck interface variant that reads from operand-token positions while still injecting the calculator result at `=`:

- `calculator_read_position=eq` as the default, preserving existing behavior.
- New diagnostic mode: `calculator_read_position=operands`, where `a_logits` come from the final A digit position and `b_logits` from the final B digit position.
- Keep output injection at `=` unchanged.
- Re-run the same `0..19` and `0..49` ladder with A/B/C/oracle controls.

This directly follows the probe evidence and is more likely to test the main thesis than further loss schedules on the current `=` readout.
