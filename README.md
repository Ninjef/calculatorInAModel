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
python scripts/diagnose_calculator_protocol.py --checkpoint runs/<run>/<child>/final_weights.pt --digits 1 --oracle --probe
python -m pytest                # runs data generator tests
```
