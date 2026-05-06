# 2026-05-05 - Low-variance action-loss continuations from selected retention checkpoints

Task: `aiAgentProjectTasks/2026-05-03-phase-2-eighth-task-Low-variance-action-loss-continuations-from-selected-retention-checkpoints.md`.

## Code changes

- Added `calculator_estimator=action_loss_replay_interface`.
- Added per-prompt replay/cache targets keyed by prompt tokens in `scripts/overfit_one_batch.py`.
- Added `--action-loss-candidate-refresh-every` and `--action-loss-candidate-ema-beta`.
- Refactored action-loss soft-target construction so the original per-step `action_loss_weighted_interface` and the replay variant share target helpers.
- Registered `action_loss_replay_interface` as an STE-like hard-action estimator in `src/model.py`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q tests/test_model.py
```

Result: `48 passed`.

## Runs

Repo-local `runs/` was not writable from this sandbox, so low-variance run artifacts were written under:

```text
/Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/runs
```

Common continuation command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator action_loss_replay_interface --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0 --action-loss-candidate-random 4 --action-loss-candidate-topk 1 --action-loss-candidate-local-radius 1 --action-loss-candidate-temperature 1.0 --action-loss-candidate-refresh-every 20 --action-loss-candidate-ema-beta 0.8 --steps 500 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 128 --log-every 50 --run-root /Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/runs
```

| Run | Start checkpoint | Extra args | Run path | Final eval exact |
| --- | --- | --- | --- | ---: |
| AL-selected-cont-seed1 | dense seed1 step `00100` | `--seed 1` | `/Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/runs/2026-05-03_175424_184826_model-c-op0-19-action_loss_replay_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-alrefresh20-alema0.8-answer_decoder/model-c-2digit-seed3` | `0.4375` |
| AL-selected-cont-seed2 | dense seed2 step `00550` | `--seed 2` | `/Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/runs/2026-05-03_175540_533678_model-c-op0-19-action_loss_replay_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-alrefresh20-alema0.8-answer_decoder/model-c-2digit-seed4` | `0.4551` |
| AL-selected-cont-seed3 | dense seed3 step `01050` | `--seed 3` | `/Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/runs/2026-05-03_175540_533811_model-c-op0-19-action_loss_replay_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-alrefresh20-alema0.8-answer_decoder/model-c-2digit-seed5` | `0.4375` |

Proof constraints from final `metrics.json`:

| Run seed | Final aux weight | Final anchor weight | Upstream frozen | Trainable groups |
| --- | ---: | ---: | --- | --- |
| seed3 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |
| seed4 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |
| seed5 | `0.0` | `0.0` | `true` | `calculator_hook.input_proj` |

## Canonical action-loss selection

Action-loss diagnostics:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --manifest-json /Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/diagnostics/action_loss_lowvar_manifest.json --samples 64 --random-actions 16 --output-root /Users/jarnold/Documents/Codex/2026-05-03/please-work-in-this-repo-users-2/diagnostics/action_loss_lowvar_selected_unique --no-work-history
```

| Run | Start gap | Best continuation checkpoint | Best gap | Operand exact | Calc result acc | Learned best |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| seed1 / run seed3 | `2.1383` | step `00000` | `2.1383` | `0.5625` | `0.5938` | `0.0000` |
| seed2 / run seed4 | `2.2018` | step `00000` | `2.2018` | `0.5469` | `0.6094` | `0.0000` |
| seed3 / run seed5 | `2.4347` | step `00450` | `2.3177` | `0.5781` | `0.5938` | `0.0000` |

The required two-of-three positive criterion failed. Stage-B-started comparisons were not run.

## Causal and private diagnostics

Causal diagnostics used:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <checkpoint> --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --output-dir <output>
```

| Checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle eval | Classification | Forced learned-best | True-sum best |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| seed1 final | `0.3438` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3438` | `0.9219` |
| seed2 final | `0.3906` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3906` | `0.9219` |
| seed3 final | `0.4375` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.4375` | `0.9219` |
| seed3 step 450 | `0.3906` | `0.0000` | `0.0156` | `0.0000` | `0.9063` | `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated` | `0.3906` | `0.9219` |

Private-protocol diagnostics used:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/diagnose_private_protocol.py --checkpoint <checkpoint> --digits 2 --operand-max 19 --output-dir <output>
```

| Checkpoint | All-pair answer | Operand exact | Calc result acc | Best affine A | Best affine B |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1 final | `0.4550` | `0.4300` | `0.4800` | `0.8000` | `0.5375` |
| seed2 final | `0.4750` | `0.4650` | `0.5025` | `0.8000` | `0.5800` |
| seed3 final | `0.4550` | `0.4150` | `0.4800` | `0.7500` | `0.5500` |
| seed3 step 450 | `0.4900` | `0.5200` | `0.5275` | `0.9500` | `0.5450` |

Group behavior for best low-variance continuation snapshot, seed3 step 450:

| Group | Count | Answer exact | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: |
| all | `400` | `0.4900` | `0.5200` | `0.5275` |
| carry | `345` | `0.4609` | `0.4957` | `0.5043` |
| no_carry | `55` | `0.6727` | `0.6727` | `0.6727` |
| large_operand | `300` | `0.4667` | `0.4933` | `0.5033` |
| small_operands | `100` | `0.5600` | `0.6000` | `0.6000` |
| symmetric | `20` | `0.6500` | `0.7000` | `0.7000` |

## Decision

Low-variance replay/EMA did not make action-loss self-training robust. It preserved the strict causal wiring and produced one transient seed3 improvement over its selected start on canonical learned-minus-true gap, but two selected continuations were best at step 0 and learned-best stayed `0.0`.

Interpretation:

- Selected checkpoints remain useful stopping points, but not reliable continuation starts for this objective.
- Candidate answer-NLL signal exists, but sampled candidate replay/EMA still does not create stable true-operand-like structure.
- This remains a no-go for upstream unfreezing.
- The next best path is to remove sampled-candidate variance entirely in the small `20 x 20` action regime by enumerating all action pairs and training on full answer-NLL soft targets before attempting upstream distillation.
