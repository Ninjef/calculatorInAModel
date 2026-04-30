# 2026-04-30 - Track 2 training-signal protocol supervision

Task: implement the Track 2 protocol-supervision plan, add the first calculator replacement bottleneck mode, and run the initial seed-0 `operand_max=19` matrix.

## Implementation

- Added `GPTConfig.calculator_injection_mode`, defaulting to `add`.
- Added `add` and `replace` application semantics in `TinyGPT`:
  - `add`: preserves the previous residual addition behavior.
  - `replace`: replaces active `=` residual positions with the calculator injection and leaves non-`=` positions unchanged.
- Threaded `--calculator-injection-mode {add,replace}` through `scripts/overfit_one_batch.py`, saved configs/checkpoints, run names, printed config, and `metrics.json`.
- Added `scripts/run_track2_protocol_supervision.py` for the Track 2 seed-0 matrix, with resume support via `--skip-first` and `--limit`.
- Fixed operand-read diagnostics during generation: if an untrained model generates an extra `=`, operand-read mode now uses the first prompt `=` as the read-position anchor instead of requiring exactly one `=`.

## Validation

```bash
PYTHONPATH=. pytest -q
python3 scripts/run_track2_protocol_supervision.py --track non-bottleneck --dry-run
python3 scripts/run_track2_protocol_supervision.py --track bottleneck --dry-run
python3 scripts/run_track2_protocol_supervision.py --track all --skip-first 4 --dry-run
```

Result:

```text
39 passed
```

The first full runner attempt stopped at the first transition-snapshot run because untrained generation produced an extra `=` under `calculator_read_position=operands`. After the fix and regression test, the matrix resumed from `--skip-first 4` and completed.

## Experiment setup

All Track 2 runs used:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
seed argument=0
run seed=2
steps=1000
eval_samples=512
```

## Non-bottleneck additive runs

| Run | Exact | Operand exact | Calc result | Final loss | Final aux loss | Counterfactuals: zero inj / forced zero / forced random / oracle eval | Path |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Answer-only Model C | `0.785` | `0.000` | `0.031` | `0.1610` | n/a | `0.828 / 0.703 / 0.758 / 0.758` | `runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2` |
| Aux `0.003` | `0.127` | `0.016` | `0.039` | `1.2691` | `26.9690` | `0.125 / 0.125 / 0.125 / 0.125` | `runs/2026-04-30_124752_856047_model-c-op0-19-aux0.003/model-c-2digit-seed2` |
| Aux `0.01` | `0.961` | `0.000` | `0.031` | `0.0900` | `3.0274` | `0.906 / 0.859 / 0.953 / 0.969` | `runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2` |
| Aux `0.03` | `0.738` | `0.000` | `0.023` | `0.2868` | `2.9364` | `0.734 / 0.641 / 0.703 / 0.750` | `runs/2026-04-30_125144_795116_model-c-op0-19-aux0.03/model-c-2digit-seed2` |
| Aux `0.03 -> 0` over 1000 | `0.018` | `0.000` | `0.008` | `1.4126` | `3515.5093` | `0.008 / 0.008 / 0.008 / 0.008` | `runs/2026-04-30_125743_434260_model-c-op0-19-aux0.03-auxdecay1000/model-c-2digit-seed2` |
| Aux `0.03 -> 0.003` over 1000 | `0.000` | `0.008` | `0.078` | `11.2679` | `3209.3799` | `0.000 / 0.000 / 0.000 / 0.000` | `runs/2026-04-30_130729_804587_model-c-op0-19-aux0.03-auxdecay1000-auxfloor0.003/model-c-2digit-seed2` |
| Aux `0.03 -> 0` over 300 | `0.004` | `0.000` | `0.055` | `1.7567` | `24241.1895` | `0.008 / 0.008 / 0.008 / 0.008` | `runs/2026-04-30_131233_089867_model-c-op0-19-aux0.03-auxdecay300/model-c-2digit-seed2` |

Retention snapshots for additive transition runs:

| Run | Snapshot readout |
| --- | --- |
| Aux `0.03 -> 0` over 1000 | Best snapshot step `200`: normal `0.078`, zero-inj `0.094`, operand `0.000`; final step `1000`: normal `0.016`, operand `0.000`. |
| Aux `0.03 -> 0.003` over 1000 | Best snapshot step `200`: normal `0.172`, zero-inj `0.172`, operand `0.000`; final step `1000`: normal `0.000`, operand `0.031`. |
| Aux `0.03 -> 0` over 300 | Best snapshot step `200`: normal `0.203`, zero-inj `0.203`, operand `0.000`; final step `1000`: normal `0.016`, operand `0.000`. |

Interpretation:

- Additive answer-only still does not learn the intended operand protocol: operand exact is `0.000`, calculator result accuracy is near chance, and counterfactuals do not show calculator dependence.
- Additive aux `0.01` gives high answer exact match, but not via true calculator use. Operand exact remains `0.000`, calculator result accuracy remains `0.031`, and injection-zero/forced-random remain high.
- Aux decay/removal does not retain a protocol. The schedules collapse, with operand exact near zero and aux losses exploding.

## Replacement-mode bottleneck runs

| Run | Exact | Operand exact | Calc result | Final loss | Final aux loss | Counterfactuals: zero inj / forced zero / forced random / oracle eval | Path |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Model B/off replacement control | `0.889` | n/a | n/a | `0.1188` | n/a | n/a | `runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2` |
| Oracle Model C replacement | `1.000` | `1.000` | `1.000` | `0.0000` | n/a | `0.039 / 0.000 / 0.047 / 1.000` | `runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2` |
| Answer-only Model C replacement | `0.133` | `0.008` | `0.047` | `0.8473` | n/a | `0.109 / 0.109 / 0.117 / 0.094` | `runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2` |
| Aux `0.003` replacement | `0.098` | `0.000` | `0.023` | `2.4433` | `462.4188` | `0.086 / 0.086 / 0.062 / 0.086` | `runs/2026-04-30_132143_624617_model-c-op0-19-replace-aux0.003/model-c-2digit-seed2` |
| Aux `0.01` replacement | `0.059` | `0.008` | `0.031` | `5.6704` | `449.2885` | `0.055 / 0.047 / 0.055 / 0.023` | `runs/2026-04-30_132351_651865_model-c-op0-19-replace-aux0.01/model-c-2digit-seed2` |
| Aux `0.03` replacement | `0.059` | `0.008` | `0.070` | `6.8333` | `189.3558` | `0.047 / 0.047 / 0.023 / 0.023` | `runs/2026-04-30_132601_196021_model-c-op0-19-replace-aux0.03/model-c-2digit-seed2` |
| Aux `0.03 -> 0` replacement | `0.094` | `0.008` | `0.086` | `1.0485` | `274.5063` | `0.047 / 0.062 / 0.055 / 0.023` | `runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2` |
| Aux `0.03 -> 0.003` replacement | `0.062` | `0.000` | `0.047` | `3.9610` | `968.9871` | `0.023 / 0.023 / 0.023 / 0.031` | `runs/2026-04-30_133322_141237_model-c-op0-19-replace-aux0.03-auxdecay1000-auxfloor0.003/model-c-2digit-seed2` |

Retention snapshots for replacement transition runs:

| Run | Snapshot readout |
| --- | --- |
| Aux `0.03 -> 0` replacement | Best snapshot step `300`: normal `0.422`, zero-inj `0.281`, oracle `0.203`, operand `0.000`, calc `0.000`; final step `1000`: normal `0.094`, operand `0.000`. |
| Aux `0.03 -> 0.003` replacement | Best snapshot step `400`: normal `0.250`, zero-inj `0.172`, oracle `0.172`, operand `0.000`, calc `0.016`; final step `1000`: normal `0.094`, operand `0.000`. |

Interpretation:

- Oracle replacement succeeds and is calculator-dependent. Zero/forced/random calculator result counterfactuals collapse while oracle-at-eval stays perfect.
- Learned replacement does not learn the intended protocol. Operand exact remains near zero and calculator result accuracy stays near chance.
- The implemented `replace` mode is not a valid calculator-required bottleneck for autoregressive generation. Model B/off reaches `0.889`, so later answer-token positions can still exploit normal residual context even though the original `=` residual is replaced.
- Treat replacement-mode learned failures as evidence against this specific soft bottleneck plus training signal, not as a clean failure of calculator-required protocol learning.

## Overall conclusion

Track 2 produced three useful results:

1. Direct oracle replacement verifies that the output-side calculator result can drive answer generation.
2. Auxiliary operand supervision in this setup did not induce or retain a stable true-operand protocol; high additive answer accuracy is mostly bypass behavior.
3. The first `replace` bottleneck is too leaky to serve as the primary calculator-required capability test.

Recommended next step: implement a stricter bottleneck that also prevents answer-token positions from attending to unreplaced operand residuals after the hook, or adds an explicit decoder phase that only receives calculator result plus minimal formatting state. Do not move to `operand_max=49` from these Track 2 results.
