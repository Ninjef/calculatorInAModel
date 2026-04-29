# 2026-04-29 — Step 8: Latent protocol diagnostics and oracle-output-side check

Task: Implement the Step 8 diagnostic plan and determine whether Model C's failure is caused by the learned operand interface, the calculator result injection/downstream path, or a wiring/training bug.

## What changed

**Calculator tracing and controls.**
- Extended `GPTConfig` with `calculator_injection_scale`, defaulting to `1.0` so existing Model A/B/C behavior is unchanged.
- Added optional diagnostics to `TinyGPT.forward(..., return_diagnostics=True)`.
- The diagnostic path records the calculator read residual and, when the hook is active, a trace containing:
  - predicted latent operands,
  - operand confidences and entropies,
  - calculator result class,
  - injection norms,
  - `=` mask/location,
  - whether oracle operands were used.
- Added `oracle_operands` support so true operands can bypass the learned input projection while preserving the calculator output projection and downstream transformer path.

**Diagnostic script.**
- Added `scripts/diagnose_calculator_protocol.py`.
- It can load saved checkpoints or train a short fresh model, emit per-example CSV rows, write JSON summaries, run oracle-input diagnostics, vary injection scale at load time, restrict operand ranges for tiny curricula, and train simple residual probes for operands A/B.
- Diagnostic outputs are saved under a checkpoint run's `diagnostics/` directory by default, or under an explicit `--output-dir`.

**Oracle-training experiment.**
- Added `--oracle-train` to `scripts/overfit_one_batch.py`.
- During training and evaluation, this feeds true operands into Model C's calculator while keeping the same calculator result embedding and downstream path.
- Added `--injection-scale` to the training script for follow-up output-side scale sweeps.
- Added tests for trace values, oracle result forcing, injection-scale behavior, diagnostic CLI smoke coverage, and fixed-width oracle operand extraction.

## Verification

Ran:

```bash
python3 -m pytest
python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-04-29_084448_model-c/model-c-1digit-seed1/final_weights.pt --digits 1 --samples 16 --operand-max 9 --oracle --probe --probe-steps 20
python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-04-29_084448_model-c/model-c-1digit-seed1/final_weights.pt --digits 1 --samples 16 --operand-max 9 --probe --probe-steps 20 --output-dir runs/2026-04-29_084448_model-c/model-c-1digit-seed1/diagnostics_learned
python3 scripts/overfit_one_batch.py --variant model-c --oracle-train --digits 1 --steps 1000 --eval-samples 256
python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-04-29_104845_model-c-oracle/model-c-1digit-seed1/final_weights.pt --digits 1 --samples 64 --operand-max 9 --oracle --probe --probe-steps 50
python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-04-29_104845_model-c-oracle/model-c-1digit-seed1/final_weights.pt --digits 1 --samples 64 --operand-max 9 --output-dir runs/2026-04-29_104845_model-c-oracle/model-c-1digit-seed1/diagnostics_learned
```

Results:
- `pytest`: 18 tests passed.
- Existing failed Model C checkpoint, learned operands:
  - exact match: `0/16 = 0.000`
  - operand exact match: `0.000`
  - calculator result accuracy: `0.000`
- Existing failed Model C checkpoint, oracle operands at eval time:
  - exact match: `0/16 = 0.000`
  - operand exact match: `1.000`
  - calculator result accuracy: `1.000`
- Oracle-trained Model C run:
  - saved to `runs/2026-04-29_104845_model-c-oracle/model-c-1digit-seed1`
  - 1-digit exact match: `236/256 = 0.922`
  - final loss: `0.0648`
- Oracle-trained checkpoint, oracle diagnostic:
  - exact match: `59/64 = 0.922`
  - operand exact match: `1.000`
  - calculator result accuracy: `1.000`
- Oracle-trained checkpoint, learned operand diagnostic:
  - exact match: `34/64 = 0.531`
  - operand exact match: `0.000`
  - calculator result accuracy: `0.094`

## Interpretation

The downstream path can learn to use calculator outputs when the calculator is trained with correct operands. This strongly suggests the basic result injection wiring is sensible:
- result one-hot -> `output_proj` -> residual stream at `=` -> downstream layers -> answer logits is a learnable path;
- the non-differentiable calculator result itself is not the bottleneck;
- the old "oracle at eval time on a failed Model C checkpoint" result was not enough to condemn the output side, because that checkpoint had already adapted to bad learned calculator inputs.

The remaining primary bottleneck is the learned operand interface:
- the input projection/discretization/STE path is not learning a useful latent operand protocol under the current training setup;
- `--oracle-train` proves the downstream side can work, while learned operands still fail even on the oracle-trained checkpoint.

## Recommended next steps

Focus Step 8 follow-up on the operand encoder rather than the result injection path:
- run tiny-vocabulary learned-operand training with operand ranges such as `0-2` and `0-4`;
- add an auxiliary operand loss as a diagnostic, not yet as the final method, to verify the input projection can learn the obvious protocol;
- compare hard STE against a soft/Gumbel-style operand interface only after tiny-vocab and auxiliary-loss checks;
- run `--injection-scale` sweeps on oracle-trained Model C only as a secondary check, since scale `1.0` is already learnable for 1-digit oracle training.

## Files changed

```text
README.md
src/model.py
scripts/overfit_one_batch.py
scripts/diagnose_calculator_protocol.py
tests/test_model.py
aiAgentWorkHistory/2026-04-29-step-8-latent-protocol-diagnostics.md
```
