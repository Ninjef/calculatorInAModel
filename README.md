# calculatorInAModel

Research sandbox for embedding a non-differentiable calculator inside a tiny decoder-only transformer. See `aiAgentProjectTasks/` for the full plan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/check_env.py     # confirms torch + MPS on M1
python scripts/sample_data.py   # prints 20 arithmetic samples
python scripts/overfit_one_batch.py   # Model A baseline
python scripts/overfit_one_batch.py --variant model-b   # calculator hook wired off
python scripts/overfit_one_batch.py --variant model-c   # latent calculator addition on
python scripts/overfit_one_batch.py --variant model-c --oracle-train --digits 1   # train with true operands fed to the calculator
python scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 4 --calculator-operand-vocab-size 5   # true tiny-vocab learned operands
python scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 4 --calculator-operand-vocab-size 5 --aux-operand-loss-weight 0.1   # diagnostic operand supervision
python scripts/overfit_one_batch.py --variant model-a --digits 1 --operand-max 9 --n-layer 1 --n-head 1 --n-embd 4 --mlp-expansion 1   # deliberately weak baseline
python scripts/overfit_one_batch.py --variant model-c --digits 1 --operand-max 9 --n-layer 1 --n-head 1 --n-embd 4 --mlp-expansion 1 --calculator-hook-after-layer 1
python scripts/diagnose_calculator_protocol.py --checkpoint runs/<run>/<child>/final_weights.pt --digits 1 --oracle --probe
python scripts/diagnose_calculator_protocol.py --checkpoint runs/<run>/<child>/final_weights.pt --digits 1 --operand-max 4 --probe --probe-layers 1 2 3 --probe-positions a b eq
python scripts/diagnose_calculator_protocol.py --checkpoint runs/<run>/<child>/final_weights.pt --digits 1 --calculator-result-override zero   # counterfactual calculator result eval
python -m pytest                # runs data generator tests
```
