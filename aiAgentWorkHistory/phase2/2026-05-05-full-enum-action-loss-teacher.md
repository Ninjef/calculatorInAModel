# 2026-05-05 - Full-enumeration action-loss teacher under frozen upstream

Task: `aiAgentProjectTasks/2026-05-05-phase-2-ninth-task-Full-action-enumeration-teacher-before-upstream-unfreezing.md`.

## Code changes

- Added `calculator_estimator=action_loss_full_enum_interface`.
- Added chunked full action-pair scoring over all `20 x 20 = 400` calculator actions.
- Added full-enum answer-NLL soft targets over action pairs, marginalized to A/B operand distributions, with no true operands/sums in target construction.
- Added knobs:
  - `--action-loss-full-enum-temperature`
  - `--action-loss-full-enum-min-probability-floor`
  - `--action-loss-full-enum-chunk-size`
- Added `scripts/run_full_enum_action_loss_diagnostic.py`.
- Added tests for full-enum marginal construction and frozen-upstream input-proj-only gradients.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q tests/test_model.py
```

Result: `50 passed`.

## Run root

Repo-local `runs/` was not writable from this sandbox, so new run artifacts were written under:

```text
/Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/runs
```

Common selected-continuation command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator action_loss_full_enum_interface --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --steps 500 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 128 --log-every 50 --run-root /Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/runs
```

Selected-checkpoint continuations:

| Run | Start checkpoint | Seed arg | Run path | Final eval exact |
| --- | --- | ---: | --- | ---: |
| FullEnum-selected-cont-seed1 | `runs/2026-05-03_112750_450950_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3/checkpoint_snapshots/step_00100_weights.pt` | `1` | `/Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/runs/2026-05-05_213924_219702_model-c-op0-19-action_loss_full_enum_interface-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed3` | `0.4824` |
| FullEnum-selected-cont-seed2 | `runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt` | `2` | `/Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/runs/2026-05-05_213924_219867_model-c-op0-19-action_loss_full_enum_interface-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed4` | `0.4297` |
| FullEnum-selected-cont-seed3 | `runs/2026-05-03_114747_345486_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed5/checkpoint_snapshots/step_01050_weights.pt` | `3` | `/Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/runs/2026-05-05_213924_220050_model-c-op0-19-action_loss_full_enum_interface-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed5` | `0.4551` |

Proof constraints from final `metrics.json`:

| Run seed | Final aux weight | Final anchor weight | Upstream frozen | Trainable groups |
| --- | ---: | ---: | --- | --- |
| seed3 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |
| seed4 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |
| seed5 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |

## Full-enum landscape diagnostic

Diagnostic command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint <checkpoints> --samples 64 --batch-size 32 --digits 2 --operand-max 19 --temperature 1.0 --chunk-size 64 --output-root /Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/diagnostics/full_enum_primary
```

For final selected continuations, full enumeration found:

| Checkpoint | Best NLL | Learned NLL | True NLL | Learned-true gap | True best | Learned best | True A mass | True B mass | Entropy | Effective pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected seed1 final | `0.0997` | `2.6643` | `0.0997` | `2.5646` | `0.0469` | `0.0312` | `0.0756` | `0.0767` | `3.2194` | `29.21` |
| selected seed2 final | `0.0997` | `2.6894` | `0.0997` | `2.5897` | `0.0469` | `0.0469` | `0.0756` | `0.0767` | `3.2194` | `29.21` |
| selected seed3 final | `0.0997` | `3.1185` | `0.0997` | `3.0189` | `0.0469` | `0.0312` | `0.0756` | `0.0767` | `3.2194` | `29.21` |

## Canonical action-loss selection

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --manifest-json /Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/diagnostics/full_enum_action_loss_manifest.json --samples 64 --random-actions 16 --output-root /Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/diagnostics/action_loss_full_enum_selected --no-work-history
```

| Run | Start gap | Best checkpoint | Best gap | Operand exact | Calc result acc | Learned best |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| selected seed1 | `2.1383` | start / step `00000` | `2.1383` | `0.5625` | `0.5938` | `0.0000` |
| selected seed2 | `2.2018` | step `00200` | `2.0201` | `0.6250` | `0.6406` | `0.0000` |
| selected seed3 | `2.4347` | step `00100` | `2.0627` | `0.5625` | `0.5938` | `0.0000` |

Two of three selected continuations improved canonical learned-minus-true gap, so the Stage-B comparison gate was met. However, `learned_best_fraction` stayed `0.0`.

## Stage-B comparison

Command stem matched the selected-continuation command except:

```bash
--semantic-decoder-checkpoint runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt --steps 1000
```

| Run | Final eval exact | Best checkpoint by action gap | Start gap | Best gap | Operand exact | Calc result acc | Learned best |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| StageB seed1 | `0.4727` | step `00600` | `3.0642` | `2.3293` | `0.5469` | `0.5625` | `0.0000` |
| StageB seed2 | `0.4707` | step `00550` | `3.0642` | `2.1961` | `0.5625` | `0.6250` | `0.0000` |
| StageB seed3 | `0.4355` | step `00500` | `3.0642` | `2.0223` | `0.5781` | `0.5938` | `0.0000` |

Stage-B starts improved by the canonical gap in all three runs, but again did not produce nonzero learned-best action-loss fraction.

## Causal diagnostics

Command template:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <checkpoint> --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --output-dir <output>
```

| Checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle eval | Classification | Forced learned-best | True-sum best |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| selected seed1 final | `0.4688` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `semantically_decodable_private_calculator_code` / `strict_bottleneck_unvalidated` | `0.4688` | `0.9219` |
| selected seed2 final | `0.3750` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3750` | `0.9219` |
| selected seed3 final | `0.4219` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.4219` | `0.9219` |
| selected seed1 start | `0.3750` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3750` | `0.9219` |
| selected seed2 step 200 | `0.3906` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3906` | `0.9219` |
| selected seed3 step 100 | `0.4219` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.4219` | `0.9219` |

## Private-protocol diagnostics

Command template:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/diagnose_private_protocol.py --checkpoint <checkpoint> --digits 2 --operand-max 19 --output-dir <output>
```

| Checkpoint | All-pair answer | Operand exact | Calc result acc | Best affine A | Best affine B |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected seed1 final | `0.4925` | `0.4900` | `0.5300` | n/a | n/a |
| selected seed1 start | `0.5025` | `0.5275` | `0.5425` | n/a | n/a |
| selected seed2 final | `0.4375` | `0.4075` | `0.4625` | n/a | n/a |
| selected seed2 step 200 | `0.5300` | `0.5675` | `0.5750` | `0.9500` | `0.5975` |
| selected seed3 final | `0.4750` | `0.4650` | `0.5075` | n/a | n/a |
| selected seed3 step 100 | `0.5075` | `0.5225` | `0.5475` | n/a | n/a |

Group behavior for best selected snapshot, selected seed2 step 200:

| Group | Count | Answer exact | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: |
| all | `400` | `0.5300` | `0.5675` | `0.5750` |
| carry | `345` | `0.5072` | `0.5507` | `0.5594` |
| no_carry | `55` | `0.6727` | `0.6727` | `0.6727` |
| large_operand | `300` | `0.5167` | `0.5533` | `0.5633` |
| small_operands | `100` | `0.5700` | `0.6100` | `0.6100` |
| symmetric | `20` | `0.7000` | `0.7500` | `0.7500` |

## Decision

Full enumeration is a useful teacher-quality diagnostic but not enough to justify upstream distillation yet.

Positive evidence:

- Full enumeration removed sampled-candidate variance and improved selected-start canonical gap in two of three runs.
- Stage-B-started comparisons improved gap in all three runs.
- Guardrails stayed intact: aux `0.0`, anchor `0.0`, upstream frozen, injection-zero near zero, oracle-at-eval high.
- Best selected private-protocol snapshot, seed2 step 200, improved all-pair operand exact to `0.5675` and calculator result accuracy to `0.5750`.

Negative evidence:

- The best selected full-enum snapshot only modestly beat the previous low-variance best on private all-pair operand exact (`0.5675` vs `0.5200`) and result accuracy (`0.5750` vs `0.5275`), but it did not preserve that through final checkpoints.
- Learned-best action-loss fraction stayed `0.0` under the canonical diagnostic for every selected and Stage-B checkpoint.
- Full-enum soft targets remain broad (`~29` effective pairs), and true A/B marginal mass is low (`~0.076`), so the teacher does not give a sharp true-operand-like target.

Recommendation: do not proceed to upstream distillation from this teacher. Treat this as evidence that answer-NLL-derived action targets alone are not enough under the current bottleneck/readout geometry. The next step should be better interface parameterization or curriculum, not upstream unfreezing.
