# 2026-04-29 - REINFORCE calculator-action experiments

Task: Implement the next upstream-learning strategy from `aiAgentProjectTasks/2026-04-29-1219-Upstream-calculator-learning-experiments.md`: sample calculator operands as stochastic actions and train the calculator input side with a score-function estimator, while preserving protocol diagnostics so answer accuracy cannot hide a broken calculator interface.

## What changed

**Sampled calculator estimator.**
- Added `GPTConfig.calculator_estimator`, with supported values `ste` and `reinforce`.
- `CalculatorHook` now samples `a` and `b` from `Categorical(logits=...)` when `calculator_estimator="reinforce"`.
- The forward pass remains hard and non-differentiable through the sampled calculator action. The trace now records `a_logp`, `b_logp`, and `sampled_logp` in addition to operand predictions, result, confidence, entropy, and injection norm.

**Policy-gradient training path.**
- Extended `scripts/overfit_one_batch.py` with:
  - `--calculator-estimator reinforce`
  - `--reinforce-baseline-beta`
  - `--reinforce-entropy-weight`
  - `--reinforce-entropy-decay-steps`
  - `--aux-operand-loss-decay-steps`
- Added per-example masked answer loss so each sampled operand pair gets its own advantage:

```text
policy_loss = mean((per_example_answer_loss.detach() - baseline) * sampled_logp_at_equals)
```

- Entropy is applied at the `=` calculator read position as a minimization loss `-entropy_weight * entropy`.
- The moving baseline is logged and updated after the optimizer step, so the curve records the actual baseline used for that training step.

**Saved diagnostics.**
- Model C runs now save `calculator_trace_rows.csv` and `diagnostic_summary.json` directly in the run directory.
- Training curves include policy-specific fields: `policy_loss`, `policy_baseline`, `policy_advantage_mean`, `sampled_logp`, `operand_entropy`, and `entropy_weight`.

## Verification

Ran:

```bash
python3 -m pytest
env PYTHONDONTWRITEBYTECODE=1 python3 -c "... one-step REINFORCE gradient check ..."
```

Results:
- Tests passed: `19 passed`.
- The gradient-plumbing check confirmed the policy term reaches the calculator input projection: `input_proj_grad_norm ~= 0.00859`, finite.
- A separate `python3 -m py_compile ...` smoke hit a sandbox write denial in `src/__pycache__`; pytest imported and exercised the same files successfully, so this was treated as a sandbox artifact rather than a code failure.

## Runs

All runs below used `--variant model-c --digits 1 --steps 1000 --batch-size 64 --eval-samples 256` unless noted.

| Run | Eval exact match | Diagnostic exact match | Operand exact match | Calc result accuracy | Mean A/B entropy | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/2026-04-29_122908_model-c-op0-1-reinforce/model-c-1digit-seed1` | `256/256 = 1.000` | `128/128 = 1.000` | `0.242` | `0.242` | `0.000 / 0.000` | Collapsed to deterministic wrong protocol on many rows. |
| `runs/2026-04-29_123735_model-c-op0-1-reinforce/model-c-1digit-seed2` | `256/256 = 1.000` | `128/128 = 1.000` | `0.000` | `0.000` | `0.000 / 0.000` | Second seed rules out an obvious lucky/unlucky single-run story. |
| `runs/2026-04-29_123105_model-c-op0-2-reinforce/model-c-1digit-seed1` | `256/256 = 1.000` | `128/128 = 1.000` | `0.141` | `0.258` | `0.765 / 1.010` | Operand distributions remain high-entropy and near chance while answers are solved. |
| `runs/2026-04-29_123258_model-c-op0-4-reinforce/model-c-1digit-seed1` | `256/256 = 1.000` | `128/128 = 1.000` | `0.055` | `0.148` | `1.609 / 1.609` | Essentially uniform over the 5-class operand vocabulary. |
| `runs/2026-04-29_123457_model-c-op0-4-reinforce-aux0.1-auxdecay500/model-c-1digit-seed1` | `256/256 = 1.000` | `128/128 = 1.000` | `0.047` | `0.086` | `0.000 / 0.000` | Aux warmup did not maintain the true protocol after decay; collapsed to a near-constant protocol. |

A short smoke run also exists at `runs/2026-04-29_122836_model-c-op0-1-reinforce/`; it used only 20 steps and is not included in the main interpretation.

## Interpretation

Vanilla single-sample REINFORCE with a moving scalar baseline does not discover the intended calculator protocol in these tiny settings. Even on `0..1`, answer accuracy reaches `1.000` while true operand exact match is poor or zero. That means the ordinary transformer path is still solving the tiny task around the calculator.

The failures are not just "the model cannot answer" and not obviously "the policy gradient is disconnected":
- answer loss trains down quickly;
- the trace/log-prob fields are populated;
- the score-function loss produces finite nonzero gradients into `calculator_hook.input_proj`.

The aux-decay result answers the maintenance question negatively for the tried schedule: a small `0.1` aux nudge decayed to zero over 500 steps did not leave behind a stable true-operand protocol. Once aux pressure disappeared, the answer path stayed good but the operand head collapsed.

## Next suggested diagnostic

The next highest-signal check is a multi-sample per-prompt diagnostic that keeps the same batch and evaluates downstream answer loss under several sampled operand pairs. If sampled calculator choices barely change the downstream answer loss after the transformer path learns the task, then the policy-gradient signal is genuinely weak. If action choices do change loss but the scalar moving baseline washes it out, use a per-example or leave-one-out multi-sample baseline before jumping to more elaborate estimators.
