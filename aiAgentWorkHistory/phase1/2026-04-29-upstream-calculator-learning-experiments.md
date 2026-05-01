# 2026-04-29 - Upstream calculator learning experiments

Task: Start Step 9 diagnostics from `aiAgentProjectTasks/2026-04-29-1059-Upstream-calculator-learning-experiments.md`, with extra care not to confuse a plumbing bug or weak metric with a real research failure.

## What changed

**True tiny-vocabulary training controls.**
- Extended `scripts/overfit_one_batch.py` with `--operand-max` so train/eval can sample restricted operands.
- Added `--calculator-operand-vocab-size` so a `0..4` run can use a real 5-class calculator interface instead of silently keeping the normal 10-class digit interface.
- Saved both fields into each run config and metrics.

**Auxiliary operand-loss diagnostic.**
- Added `--aux-operand-loss-weight` for Model C.
- The loss is training-only cross entropy from the calculator input projection at the `=` position to true operand A/B.
- The normal answer loss is still logged separately as `answer_loss`, so answer learning and protocol learning can be compared.

**Probe and trace improvements.**
- `TinyGPT.forward(..., return_diagnostics=True)` now records residual snapshots after every transformer layer.
- `scripts/diagnose_calculator_protocol.py` can sweep `--probe-layers` and `--probe-positions a b eq`.
- Diagnostic summaries now include mean operand confidence and entropy, not just exact-match rates.

## Verification

Ran:

```bash
python3 -m pytest
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 1 --calculator-operand-vocab-size 2 --steps 1000 --eval-samples 256 --log-every 100
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 2 --calculator-operand-vocab-size 3 --steps 1000 --eval-samples 256 --log-every 100
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 4 --calculator-operand-vocab-size 5 --steps 1000 --eval-samples 256 --log-every 100
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 4 --calculator-operand-vocab-size 5 --aux-operand-loss-weight 0.1 --steps 1000 --eval-samples 256 --log-every 100
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 9 --calculator-operand-vocab-size 10 --aux-operand-loss-weight 0.1 --steps 1000 --eval-samples 256 --log-every 100
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 9 --calculator-operand-vocab-size 10 --aux-operand-loss-weight 1.0 --steps 1000 --eval-samples 256 --log-every 100
```

All tests passed: `18 passed`.

## Results

| Run | Eval exact match | Diagnostic exact match | Operand exact match | Calc result accuracy | Mean A/B confidence | Mean A/B entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/2026-04-29_111208_model-c-op0-1` | `256/256 = 1.000` | `64/64 = 1.000` | `0.203` | `0.203` | `1.000 / 1.000` | `0.001 / 0.000` |
| `runs/2026-04-29_111340_model-c-op0-2` | `256/256 = 1.000` | `64/64 = 1.000` | `0.375` | `0.438` | `0.952 / 0.998` | `0.127 / 0.014` |
| `runs/2026-04-29_111456_model-c-op0-4` | `256/256 = 1.000` | `64/64 = 1.000` | `0.000` | `0.000` | `1.000 / 1.000` | `0.000 / 0.000` |
| `runs/2026-04-29_111618_model-c-op0-4-aux0.1` | `256/256 = 1.000` | `64/64 = 1.000` | `0.938` | `0.938` | `0.963 / 0.965` | `0.124 / 0.082` |
| `runs/2026-04-29_111756_model-c-op0-9-aux0.1` | `221/256 = 0.863` | `50/64 = 0.781` | `0.141` | `0.141` | `0.835 / 0.902` | `0.374 / 0.226` |
| `runs/2026-04-29_111935_model-c-op0-9-aux1` | `17/256 = 0.066` | `2/64 = 0.031` | `0.016` | `0.063` | `1.000 / 1.000` | `0.000 / 0.000` |

Probe sweep highlights:
- On `0..1`, the current layer-2 `=` residual linearly probes both operands at `1.000 / 1.000`, even though the hook's own operand exact match is only `0.203`.
- On `0..2`, layer-2 `=` probes both operands at `1.000 / 1.000`, while the hook reaches only `0.375` operand exact match.
- On `0..4`, layer-2 `=` probes reached `0.846 / 0.923`, while the hook reaches `0.000`.
- With `0..4` auxiliary weight `0.1`, layer-2 `=` probes are `0.923 / 1.000` and the hook itself reaches `0.938`.

Representative trace rows:
- Without aux on `0..4`, the model answers correctly while sending confident wrong calculator inputs, for example `4+2=` sends `a_pred=3, b_pred=2, result=5` and still predicts `6<eos>`.
- With aux on `0..4`, rows mostly become the intended protocol, for example `4+2=` sends `a_pred=4, b_pred=2, result=6`.

## Interpretation

The smallest answer range learned by current hard STE is at least `0..4`, but this is misleading because the calculator protocol is not learned. The model can solve tiny answer distributions while routing around the calculator or using a non-human latent shortcut; protocol exact match is the metric that matters here.

The auxiliary diagnostic makes the `0..4` protocol learnable (`0.000` to `0.938` operand exact match), so the residual stream and input projection can represent the operands. This points away from missing residual information and toward the downstream answer loss being too indirect for protocol discovery.

For full `0..9`, auxiliary supervision needs more care. Weight `0.1` improves answer accuracy but does not make the learned operands reliable, while weight `1.0` diverges with the current optimizer/learning rate. The next best diagnostic is a gentler aux schedule or lower LR for the input-side objective before moving on to estimator variants.

## Files changed

```text
README.md
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
aiAgentWorkHistory/2026-04-29-upstream-calculator-learning-experiments.md
```
