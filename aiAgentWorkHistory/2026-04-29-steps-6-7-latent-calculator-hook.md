# 2026-04-29 — Steps 6-7: Latent calculator hook

Task: Implement Model B and Model C from the research plan. The calculator is an internal model component driven by residual-stream activations, not a generated tool call or token-level parser.

## What changed

**Latent calculator hook.**
- Extended `GPTConfig` with calculator settings for enabling the hook, choosing `off` vs `add`, selecting the hook layer, and sizing operand/result vocabularies.
- Added `CalculatorHook` in `src/model.py`.
- The hook reads residual activations, projects them to two latent operand logits, hard-discretizes them, computes an internal sum in `model-c`, projects the result back to the residual dimension, and adds the injection only at `=` token positions.
- The `=` token is only the internal read/write location. The calculator does not parse raw input text or generated output tokens.
- Added `HardAddSTE`, a custom autograd function whose forward pass returns the hard sum class and whose backward routes result-class gradients back to possible operand logits.

**Model variants.**
- Updated `scripts/overfit_one_batch.py` with `--variant model-a|model-b|model-c`.
- `model-a`: raw transformer baseline.
- `model-b`: calculator hook wired in but returning zero injection.
- `model-c`: latent calculator addition enabled.
- Saved run configs, metrics, checkpoints, curves, and summaries now include the variant and calculator config.

**Tests and docs.**
- Added tests for the off hook, hard-add forward pass, STE backward routing, localized `=` injection, and causal masking with the calculator enabled.
- Updated the README with the new variant commands.

**Future-work plan refinement.**
- Updated `aiAgentProjectTasks/2026-04-27-1828-Steps-next.md` after discussing the Model C failure mode.
- Added a 2026-04-29 refinement stating that the central research object is the latent calculator language, not just the presence of a hard calculator in the forward pass.
- Refined the next phase around diagnostics before architecture changes:
  - log true operands vs latent operands sent to the calculator,
  - run an oracle-input calculator experiment,
  - probe whether residual vectors already contain operand information,
  - try tiny operand vocabularies before full 1-digit addition,
  - test calculator injection strength.
- Updated the longer plan text so future Step 8/9 work is driven by specific failure-mode questions rather than broad ablation hunting.

## Verification

Ran:

```bash
python3 -m pytest
python3 scripts/overfit_one_batch.py --variant model-b --digits 1 --steps 20
python3 scripts/overfit_one_batch.py --variant model-c --digits 1 --steps 20
python3 scripts/overfit_one_batch.py --variant model-b
python3 scripts/overfit_one_batch.py --variant model-c
```

Results:
- `pytest`: 13 tests passed.
- Short Model B/C smoke runs completed and saved run directories.
- Full Model B run saved to `runs/2026-04-29_083546_model-b`.
- Full Model C run saved to `runs/2026-04-29_084448_model-c`.

Full-run exact-match:

```text
Model B:
1-digit: 251/256 = 0.980, final loss 0.0581
2-digit:  24/256 = 0.094, final loss 0.8049
3-digit:   3/256 = 0.012, final loss 1.0732

Model C:
1-digit:  17/256 = 0.066, final loss 1.6177
2-digit:   0/256 = 0.000, final loss 1.8488
3-digit:   0/256 = 0.000, final loss 1.9200
```

## Interpretation

Step 6 passed: Model B behaves like a sane wired-off control and broadly matches or slightly exceeds prior Model A baseline results under the default budget.

Step 7 is implemented, but this first Model C run is worse than Model B. That means the active calculator path is currently harmful or not learning a useful latent operand interface. The next move should be Step 8 diagnostics: log the operand IDs/probabilities the hook is feeding to the calculator on held-out problems and inspect whether the encoder is producing meaningful operands or garbage.

## Files changed

```text
README.md
src/model.py
scripts/overfit_one_batch.py
tests/test_model.py
aiAgentWorkHistory/2026-04-29-steps-6-7-latent-calculator-hook.md
aiAgentProjectTasks/2026-04-27-1828-Steps-next.md
```
