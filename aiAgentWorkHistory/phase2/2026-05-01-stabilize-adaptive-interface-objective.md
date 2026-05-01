# 2026-05-01 - Stabilize adaptive calculator interface objective

Task: `aiAgentProjectTasks/2026-05-01-phase-2-second-task-Stabilize-adaptive-interface-objective.md`

## Implementation

- Added adaptive-interface optimizer groups in `scripts/overfit_one_batch.py`:
  - `--input-proj-lr` for `calculator_hook.input_proj`.
  - `--upstream-lr` for other trainable non-frozen parameters.
  - Frozen `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` remain excluded.
- Added `--adaptive-interface-target-mode {hard_pair,soft_result}`.
  - `hard_pair` preserves the previous best-compatible-pair CE rule.
  - `soft_result` trains total probability mass over all operand pairs whose sum matches the downstream-selected result class.
- Added `--adaptive-interface-entropy-weight` as an entropy bonus on calculator input operand distributions.
- Logged target mode, optimizer-group LRs, adaptive entropy, adaptive target loss, and adaptive objective in config/metrics/training curves.
- Added unit coverage for soft-result mass loss, entropy gradients, optimizer grouping, and CLI persistence.

Validation commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_track4_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
```

Result: `52 passed`.

## Shared Starting Regime

Semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

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
calculator_read_position=operands
calculator_bottleneck_mode=answer_decoder
calculator_estimator=adaptive_interface
steps=1000
batch_size=64
eval_samples=512
seed=0
```

Frozen in all adaptive runs: `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder`.

Base training command:

```bash
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 1000 --batch-size 64 --eval-samples 512 --seed 0 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt --snapshot-every 200 --snapshot-samples 64 --log-every 50
```

Run-specific modifiers and outputs:

| Run | Modifiers | Output |
| --- | --- | --- |
| Frozen upstream | `--freeze-upstream-encoder` | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` |
| Lower LR | `--input-proj-lr 3e-4 --upstream-lr 3e-4` | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` |
| Soft result | `--adaptive-interface-target-mode soft_result` | `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-soft_result-answer_decoder/model-c-2digit-seed2` |
| Entropy 0.001 | `--input-proj-lr 3e-4 --upstream-lr 3e-4 --adaptive-interface-entropy-weight 0.001` | `runs/2026-05-01_081215_279041_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.001-answer_decoder/model-c-2digit-seed2` |
| Entropy 0.003 | `--input-proj-lr 3e-4 --upstream-lr 3e-4 --adaptive-interface-entropy-weight 0.003` | `runs/2026-05-01_081215_279020_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.003-answer_decoder/model-c-2digit-seed2` |
| Entropy 0.01 | `--input-proj-lr 3e-4 --upstream-lr 3e-4 --adaptive-interface-entropy-weight 0.01` | `runs/2026-05-01_081215_283303_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-ient0.01-answer_decoder/model-c-2digit-seed2` |

Track 3 command template, run for each checkpoint:

```bash
python3 scripts/diagnose_calculator_protocol.py --checkpoint <run>/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir <run>/track3_diagnostics
```

Track 4 command template, run for each checkpoint:

```bash
PYTHONPATH=. python3 scripts/run_track4_action_loss_diagnostic.py --checkpoint <run>/final_weights.pt --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

Note: Track 4 was first attempted without `PYTHONPATH=.` and failed on the wrapper import; rerunning with `PYTHONPATH=.` matched the task command. The reruns needed escalated filesystem access because Track 4 creates a new output directory under each run.

## Results

Baseline failed adaptive run for comparison:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
```

Baseline key metrics: eval exact `0.0098`, operand exact `0.0000`, calculator result accuracy `0.0156`, adaptive target result accuracy `0.9375`, learned-target agreement `0.0156`, final adaptive CE `193.0297`, Track 3 `calculator_ignored_or_bypassed`, Track 4 learned-minus-true gap `14.3265`.

| Run | Eval exact | Diagnostic exact | Operand exact | Calc result acc | Target result acc | Learned-target agreement | Final adaptive CE | Final entropy | Track 3 class | Track 3 learned best | Track 4 learned-true gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Frozen upstream | `0.1016` | `0.1016` | `0.0234` | `0.1016` | `0.9375` | `0.0625` | `2.8654` | `5.9605` | `causally_useful_opaque_private_code` | `0.0781` | `9.0455` |
| Lower LR | `0.0664` | `0.0469` | `0.0156` | `0.0469` | `0.9375` | `0.0703` | `2.2887` | `4.9232` | `causally_useful_opaque_private_code` | `0.0781` | `10.4115` |
| Soft result | `0.0059` | `0.0156` | `0.0000` | `0.0156` | `0.9375` | `0.0078` | `226.3104` | `0.0000` | `calculator_ignored_or_bypassed` | `0.0312` | `8.4456` |
| Entropy 0.001 | `0.0410` | `0.0625` | `0.0078` | `0.0625` | `0.9375` | `0.0625` | `2.3270` | `4.9832` | `causally_useful_opaque_private_code` | `0.0625` | `10.2226` |
| Entropy 0.003 | `0.0254` | `0.0469` | `0.0000` | `0.0469` | `0.9375` | `0.0234` | `2.3133` | `4.8528` | `calculator_ignored_or_bypassed` | `0.0469` | `9.2909` |
| Entropy 0.01 | `0.0391` | `0.0469` | `0.0000` | `0.0469` | `0.9375` | `0.0859` | `2.3345` | `5.0526` | `calculator_ignored_or_bypassed` | `0.1094` | `6.9002` |

All runs preserved oracle-at-eval and showed counterfactual degradation relative to oracle. Built-in 128-sample counterfactuals:

| Run | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval |
| --- | ---: | ---: | ---: | ---: |
| Frozen upstream | `0.0078` | `0.0000` | `0.0156` | `0.9531` |
| Lower LR | `0.0078` | `0.0000` | `0.0156` | `0.9531` |
| Soft result | `0.0078` | `0.0000` | `0.0156` | `0.9531` |
| Entropy 0.001 | `0.0078` | `0.0000` | `0.0156` | `0.9531` |
| Entropy 0.003 | `0.0078` | `0.0000` | `0.0156` | `0.9531` |
| Entropy 0.01 | `0.0078` | `0.0000` | `0.0156` | `0.9531` |

Track 3 oracle-at-eval was `0.90625` for all six runs, and true-sum forced class was best on `0.921875` of prompts for all six runs. Track 4 true actions remained far better than random/shuffled in all runs: mean true-action NLL `0.1181`, mean random-action NLL `9.9763`, mean shuffled-action NLL `10.4317`.

## Parameter Movement

L2 / max-abs delta versus the oracle semantic decoder checkpoint:

| Run | `input_proj.weight` | `input_proj.bias` | `output_proj.weight` | `answer_offset_emb` | `answer_decoder` |
| --- | --- | --- | --- | --- | --- |
| Frozen upstream | `16.1956 / 1.3765` | `5.8333 / 1.3898` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Lower LR | `1.6826 / 0.1020` | `0.4181 / 0.0921` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Soft result | `13.1783 / 0.7299` | `2.6397 / 0.5671` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Entropy 0.001 | `1.6908 / 0.1097` | `0.4195 / 0.0992` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Entropy 0.003 | `1.7149 / 0.1142` | `0.4259 / 0.1012` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Entropy 0.01 | `1.5366 / 0.1077` | `0.3858 / 0.0972` | `0.0000 / 0.0000` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |

## Interpretation

Useful negative result, with one narrow improvement:

- Frozen-upstream improved answer/calculator result accuracy over the failed adaptive baseline (`0.1016` eval, `0.1016` calculator result accuracy) and Track 4 learned-minus-true gap improved from `14.3265` to `9.0455`, but operand exact remained very low and learned actions were never Track 4 best.
- Lower LR reduced parameter movement and kept CE small, but Track 3 showed a constant learned result class (`17`) and Track 4 learned-best fraction stayed `0.0`.
- Soft-result target did not fix hard-pair collapse. It collapsed to a constant result class (`36`) with near-zero entropy and very high final adaptive loss.
- Entropy on the lower-LR setup did not rescue stable learned actions. The `0.01` run improved Track 4 learned-minus-true gap to `6.9002` and Track 3 learned-best fraction to `0.1094`, but still had operand exact `0.0`, Track 4 learned-best `0.0`, and Track 3 classified it as `calculator_ignored_or_bypassed`.

Go/no-go recommendation: no-go on claiming adaptive-interface success. The counterfactual downstream result signal still reliably identifies the correct result class, and the strict semantic decoder remains healthy, but the learned interface does not acquire stable true or consistently useful calculator actions. Frozen-upstream being better than trainable-upstream suggests co-adaptation is destabilizing, while soft-result collapse shows that removing the hard-pair tie-break alone is insufficient. The next round should keep strict bottleneck gates but consider an explicitly slower or staged interface objective, target smoothing/clipping, or a small true-operand auxiliary stabilizer as a diagnostic rather than as success evidence.
