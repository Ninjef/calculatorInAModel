# 2026-04-29 - Dumber model calculator-reliance diagnostics

Task: Work on `aiAgentProjectTasks/2026-04-29-1245-Dumber-model-calculator-reliance.md`, with extra care to avoid mistaking bugs or measurement artifacts for a research failure.

## What changed

**Tiny architecture controls.**
- Added `GPTConfig.mlp_expansion`.
- Added training/diagnostic CLI flags:
  - `--n-layer`
  - `--n-head`
  - `--n-embd`
  - `--mlp-expansion`
  - `--calculator-hook-after-layer`
- Kept existing defaults equivalent to the old 4-layer, 4-head, 128-dim, 4x-MLP setup.
- Made the default calculator hook placement legal for 1-layer runs: layer 2 for depth >= 2, otherwise layer 1.

**Experiment hygiene fixes.**
- Fixed a Model A/B/C initialization confound: calculator hook modules were previously created before the final LM head and consumed RNG during `self.apply`, so Model B with the hook off did not share the same core initialization as Model A.
- Added tests that Model B's non-calculator core initializes identically to Model A and that calculator-off forward logits exactly match the no-hook model.
- Hardened run directory creation so parallel sweeps cannot collide when launched in the same timestamp window.

**Docs/tests.**
- Added README examples for deliberately weak tiny-model runs.
- Added tests for MLP expansion and 1-layer calculator configs.

## Verification

Ran:

```bash
python3 -m pytest
```

Result: `23 passed`.

## Tiny baseline sweep

All runs used 1-digit addition, operands `0..9`, `1000` steps, batch size `64`, eval samples `512`, and MLP expansion `1`.

| Variant | Architecture | Seed | Params | Eval exact match | Final loss | Run |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Model A | 1 layer, 2 heads, 32 embd | 1 | 7,616 | `1.000` | `0.0000` | `runs/2026-04-29_125219_model-a-op0-9/model-a-1digit-seed1` |
| Model A | 1 layer, 1 head, 16 embd | 1 | 2,272 | `0.967` | `0.0288` | `runs/2026-04-29_125431_202439_model-a-op0-9/model-a-1digit-seed1` |
| Model A | 1 layer, 1 head, 8 embd | 1 | 752 | `1.000` | `0.0064` | `runs/2026-04-29_125257_model-a-op0-9/model-a-1digit-seed1` |
| Model A | 1 layer, 1 head, 4 embd | 1 | 280 | `0.691` | `0.4019` | `runs/2026-04-29_125633_803034_model-a-op0-9/model-a-1digit-seed1` |
| Model A | 1 layer, 1 head, 4 embd | 2 | 280 | `0.463` | `0.5082` | `runs/2026-04-29_125801_013436_model-a-op0-9/model-a-1digit-seed2` |

Interpretation: The candidate `1x32` architecture is still far too capable for the intended diagnostic. A truly constrained baseline was only found after shrinking to `n_embd=4`, which is below the original candidate list.

## Matching Model B control

| Variant | Architecture | Seed | Params | Eval exact match | Final loss | Run |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Model B | 1 layer, 1 head, 4 embd, hook off after layer 1 | 1 | 456 | `0.588` | `0.4392` | `runs/2026-04-29_125706_439698_model-b-op0-9/model-b-1digit-seed1` |
| Model B | 1 layer, 1 head, 4 embd, hook off after layer 1 | 2 | 456 | `0.490` | `0.5062` | `runs/2026-04-29_125801_013448_model-b-op0-9/model-b-1digit-seed2` |

After the initialization fix, Model B starts from the same initial answer loss as Model A and is in the same performance band. The remaining differences look like normal tiny-model optimization variance rather than hook plumbing.

## Model C diagnostics

| Variant | Seed | Eval exact match | Diagnostic exact match | Operand exact match | Calc result accuracy | Mean A/B entropy | Run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Model C STE | 1 | `0.418` | `0.414` | `0.008` | `0.094` | `2.293 / 2.288` | `runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1` |
| Model C STE | 2 | `0.916` | `0.867` | `0.023` | `0.117` | `1.916 / 1.922` | `runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2` |
| Model C oracle operands | 1 | `0.975` | `0.977` | `1.000` | `1.000` | `2.303 / 2.303` | `runs/2026-04-29_125931_512640_model-c-oracle-op0-9/model-c-1digit-seed1` |
| Model C STE + aux `0.1` | 1 | `0.939` | `0.898` | `0.008` | `0.008` | `2.223 / 2.232` | `runs/2026-04-29_125931_512642_model-c-op0-9-aux0.1/model-c-1digit-seed1` |

Interpretation:
- Oracle operands work, so the calculator result/output injection side is usable even in the 4-dim model.
- Plain STE can produce high answer accuracy on one seed, but the calculator traces remain wrong and high-entropy under the intended true-operand metric. The counterfactual section below shows this is not pure bypass: the model is using a non-human calculator code.
- Constant auxiliary weight `0.1` improves answer accuracy but does not teach the true operand protocol in this extreme architecture.

## Residual probes

Layer-1 `=` residual probes with 256 samples and 200 probe steps:

| Checkpoint | Operand A probe | Operand B probe |
| --- | ---: | ---: |
| Model A seed 1 | `0.173` | `0.154` |
| Model B seed 1 | `0.154` | `0.192` |
| Model C seed 2 | `0.212` | `0.308` |

Interpretation: The 4-dim architecture is small enough that Model A/B struggle, but it may also be too narrow to make the operands linearly accessible at the hook. This means failure of learned calculator reliance here should not be over-interpreted as a pure STE/protocol-discovery failure.

## Current conclusion

Reducing capacity does expose a useful diagnostic regime, but the first architecture where A/B struggle (`1 layer, 1 head, 4 embd`) is probably too weak to be the clean calculator-reliance testbed. The better next target is between the extremes:

- `1 layer, 1 head, 8 embd` with a lower step budget or harder train/eval split, because it can parse but currently solves `0..9`;
- or `1 layer, 1 head, 4 embd` with more probe positions/layers unavailable here, different injection scales, and lower-LR auxiliary schedules, if the aim is to squeeze this minimal setting.

The initial research finding from this pass was that answer gains in tiny Model C still do not imply intended true-operand calculator use, and oracle success shows the output side is not the blocking bug. The counterfactual follow-up below refines this: answer gains can still reflect a useful non-human calculator protocol.

## Counterfactual calculator-use follow-up

Follow-up question: Could low operand exact match still hide a useful non-human calculator protocol?

Answer from causal eval: yes. At least one tiny Model C checkpoint is using the calculator path in a way that helps answer accuracy, even though it is not using the intended true-operand protocol.

### What changed

Added eval-only calculator result counterfactuals:

- `--calculator-result-override add`: normal learned operands plus hard add.
- `--calculator-result-override zero`: replace the calculator result class with zero.
- `--calculator-result-override plus_one`: replace the result with `(a_pred + b_pred + 1) % result_vocab`.
- `--calculator-result-override random`: replace the result with a random result class.
- `--injection-scale 0`: leave traces intact but remove calculator residual injection.
- `--oracle`: feed true operands into the existing trained checkpoint.

These modes are implemented in `CalculatorHook`/`TinyGPT.forward` and exposed by `scripts/diagnose_calculator_protocol.py`.

### Verification

Ran:

```bash
python3 -m pytest
```

Result: `24 passed`.

### Strong weird-protocol checkpoint

Checkpoint: `runs/2026-04-29_125836_910885_model-c-op0-9-1/model-c-1digit-seed2/final_weights.pt`

All rows below use 512 eval samples.

| Eval condition | Exact match | Operand exact match | Calc result accuracy |
| --- | ---: | ---: | ---: |
| Normal learned calculator result | `0.904` | `0.014` | `0.146` |
| Injection scale `0` | `0.645` | `0.014` | `0.146` |
| Result forced to zero | `0.605` | `0.014` | `0.010` |
| Result shifted by plus-one | `0.578` | `0.014` | `0.119` |
| Random result class | `0.650` | `0.014` | `0.064` |
| Oracle true operands | `0.637` | `1.000` | `1.000` |

Interpretation:

- Removing or corrupting the calculator residual sharply hurts answer accuracy.
- Feeding the true operands also hurts, which means this checkpoint is not merely close to the intended human-readable protocol.
- The model appears tuned to its own learned calculator-result code.

### Lower-performing checkpoint sanity check

Checkpoint: `runs/2026-04-29_125836_910885_model-c-op0-9/model-c-1digit-seed1/final_weights.pt`

| Eval condition | Exact match | Operand exact match | Calc result accuracy |
| --- | ---: | ---: | ---: |
| Normal learned calculator result | `0.418` | `0.016` | `0.086` |
| Injection scale `0` | `0.361` | `0.016` | `0.086` |
| Result forced to zero | `0.361` | `0.016` | `0.010` |
| Random result class | `0.314` | `0.016` | `0.064` |

This weaker seed shows the same direction but a smaller absolute effect.

### Result-code analysis

For the strong checkpoint's normal learned calculator result:

- The calculator emitted only three result classes on 512 samples: `{4: 32, 6: 109, 11: 371}`.
- Mutual information between calculator result class and true sum was about `1.06` bits.
- A best single mapping from result class to true sum only reached `0.256` accuracy, so the code is helpful but not simply a one-token sum label.

### Revised conclusion

The earlier "bypass/shortcut" wording was too strong. A better statement is:

Model C is not learning the intended true-operand calculator protocol, but at least one tiny checkpoint does learn to causally depend on the calculator output as a non-human latent code that improves answer accuracy.

This is actually more interesting than a pure bypass, and the next diagnostic should study the learned code directly: counterfactual result-class sweeps per prompt, mapping result classes to answer logits, and training pressure that distinguishes "useful latent protocol" from "human-readable operand protocol."
