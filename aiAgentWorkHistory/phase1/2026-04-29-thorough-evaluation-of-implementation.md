# 2026-04-29 - Thorough implementation evaluation

Task: Review the functional codebase against `aiAgentProjectTasks/2026-04-29-2120-thorough-evaluation-of-implementation.md`, looking for gross logical errors that could make recent calculator-protocol conclusions untrustworthy.

## Verdict

The core implementation is broadly true to the project vision:

- Model A is a plain tiny causal transformer.
- Model B wires the calculator hook but leaves it off, and tests verify its core initialization and forward logits match the no-hook model.
- Model C injects a hard calculator result additively into the residual stream at the configured hook layer.
- The training loss remains ordinary next-token answer loss unless explicit diagnostic controls are enabled.
- Oracle operands, auxiliary operand supervision, residual probes, injection-scale controls, counterfactual calculator-result overrides, and REINFORCE traces are implemented as diagnostics rather than silently changing the default experiment.

I did not find a serious implementation bug that invalidates the main recent conclusions:

- Answer accuracy alone is not sufficient evidence of calculator-protocol learning.
- Oracle-trained Model C shows the output/injection side can be learned.
- Hard STE and single-sample REINFORCE often fail to learn the intended true-operand protocol under downstream answer loss.
- Some tiny Model C checkpoints causally depend on a non-human calculator-result code, so "pure bypass" is too strong for those cases.

## Issue fixed

Found one real hook-placement footgun:

- `GPTConfig` and CLI validation allowed `calculator_hook_after_layer=0`.
- `TinyGPT.forward` previously only checked hook insertion inside the numbered transformer-block loop, so layer `0` silently meant "calculator enabled in config, but never called."
- This did not appear to affect the documented recent runs, which used hook positions after layer 1 or later, but it could have invalidated future placement sweeps.

Fix:

- Added an embedding-stream hook path before the block loop when `calculator_hook_after_layer == 0`.
- Added a regression test that forces known calculator operands through the layer-0 hook and verifies the trace/result/injection.

Files changed:

```text
src/model.py
tests/test_model.py
```

## Review notes

Data/task generation:

- Fixed-width arithmetic sequences, loss masks, padding, and target shifting are internally consistent.
- The answer loss starts after `=`, including `<eos>`, matching the intended next-token arithmetic setup.
- Restricted operand-range runs correctly pair `--operand-max` with `--calculator-operand-vocab-size` validation.

Calculator hook:

- Injection is additive and localized to `=` positions.
- The calculator result can still affect later answer positions through remaining causal layers and attention.
- The off-mode hook returns zero injection, and tests verify Model B forward equivalence to Model A.
- Oracle operands bypass the learned input projection while preserving the calculator output projection and downstream path, which is the correct output-side diagnostic.

Diagnostics/measurement:

- Trace rows are taken at the `=` read position, which is the correct place to measure the calculator query.
- Diagnostic generation is autoregressive and trims at `<eos>`, matching the user-facing answer metric.
- Probe diagnostics use detached residual snapshots, so they inspect representation without changing the model.
- Counterfactual result overrides and injection-scale zero are appropriate causal-use checks.

REINFORCE path:

- The score-function term uses per-example answer loss and sampled log-probability at the `=` action.
- The sign is consistent with minimizing expected answer loss.
- Entropy regularization is applied as a bonus via a negative entropy loss.
- Current REINFORCE evaluation remains stochastic because the hook samples in eval mode; this is acceptable for measuring the sampled policy, but future writeups should be explicit when reporting single-sample eval metrics.

## Remaining risks

These are research/measurement caveats, not discovered code-breaking bugs:

- Tiny stochastic-policy evals should ideally report multiple eval seeds or argmax-policy diagnostics before making fine-grained claims.
- Very small architectures, especially `n_embd=4`, may be too narrow for clean operand linear accessibility, so failures there should not be overgeneralized.
- The hard STE backward is a structured surrogate through possible sums, not a literal identity-through-rounding estimator. That is defensible, but future comparisons should name it precisely.
- Existing run directories include many exploratory attempts with similar names; important conclusions should keep citing exact run paths/checkpoints.

## Verification

Ran:

```bash
python -m pytest
```

Result after the hook-layer fix:

```text
25 passed
```
