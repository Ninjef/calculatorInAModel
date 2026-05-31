# 2026-05-31 Online Hard Memory Semantic-Distill Routed Shared Output

## Question

Can the less-prescriptive online-hard-memory plus additive semantic-distillation
source survive a many-calculator/shared-output stress?

This is a thesis-relevant scaling gate rather than a local hyperparameter
tweak. The earlier hard-assignment routed branch showed that four cloned-output
hooks can train and transfer, while shared output projection trained the source
but missed trusted handoff. The hypothesis here is that arbitrary-result
semantic distillation supplies the missing shared-output transfer geometry.

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_src800/2026-05-30_201930_719833_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-c00fadadaf/model-c-2digit-seed6
```

Settings:

- op19 exhaustive `400`-prompt grid.
- Four calculator hooks with `left_operand_mod` routing.
- `--share-calculator-output-proj`.
- Sparse zero-improvement online hard memory with topk8+unique24 candidates.
- Additive semantic distillation weight `1`, sample count `8`.
- Source bottleneck mode `answer_decoder`.

Results:

| Step | Normal | Zero-inj | Calc | Distill agree | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0200 | 0.0575 | 0.0200 | 0.0340 | active |
| 200 | 0.6950 | 0.0425 | 0.6950 | 0.5420 | 96,000 |
| 400 | 1.0000 | 0.0475 | 1.0000 | 0.7050 | 96,000 |
| 600 | 1.0000 | 0.0475 | 1.0000 | 0.7416 | 96,000 |
| 800 | 1.0000 | 0.0450 | 1.0000 | 0.8069 | 96,000 |

Final source metrics:

- Final eval: `400/400 = 1.0000`.
- Diagnostic calculator-result accuracy: `1.0000`.
- Final 128-sample counterfactuals: injection-zero `0.0391`,
  forced-zero `0.0000`, forced-random `0.0391`.
- Final routed diagnostic: all four routed hooks reached calculator-result
  accuracy `1.0000`.

## Trusted Handoff

```text
runs/ohm_semdist_hooks4_shareout_handoff600/2026-05-30_202214_356789_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed6
```

Settings:

- Loaded source final checkpoint with `compatible_model`.
- Additive non-bottleneck mode: `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `--freeze-semantic-decoder`.
- `--freeze-calculator-policy`.
- Four routed hooks with shared output projection preserved.
- `600` downstream/readout steps.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3225 | 0.0275 | 0.0000 | 0.0175 | 1.0000 |
| 100 | 1.0000 | 0.0125 | 0.0075 | 0.0200 | 1.0000 |
| 200 | 1.0000 | 0.0300 | 0.0050 | 0.0125 | 1.0000 |
| 300 | 1.0000 | 0.0300 | 0.0000 | 0.0275 | 1.0000 |
| 400 | 1.0000 | 0.0125 | 0.0050 | 0.0150 | 1.0000 |
| 500 | 1.0000 | 0.0200 | 0.0025 | 0.0275 | 1.0000 |
| 600 | 1.0000 | 0.0325 | 0.0050 | 0.0175 | 1.0000 |

Final handoff metrics:

- Final eval: `400/400 = 1.0000`.
- Final 128-sample counterfactuals: injection-zero `0.0391`,
  forced-zero `0.0000`, forced-random `0.0391`.
- Diagnostic calculator-result accuracy: `1.0000`.
- All four routed hooks had calculator-result accuracy `1.0000` in the final
  routed diagnostic summary.

## Interpretation

This is a positive many-calculator/shared-output geometry result for the
less-prescriptive branch. The same shared-output architectural constraint that
missed trusted handoff under the hard-assignment routed recipe clears when the
source is trained with online hard memory plus arbitrary-result semantic
distillation.

The result does not complete the thesis. It is still an op19 fixed-grid source,
uses per-prompt online memory, uses sparse forced-result scoring before memory
fills, and reuses the seed lineage where the single-hook semantic-distilled
source was handoff-friendly. The next useful tests are fresh routed/shared
seeds, streaming/fresh-prompt memory, or larger-range routed/shared stress. Do
not repeat this exact four-hook shared-output seed as novelty.

## Decision

```text
online_hard_memory_semantic_distill_shared_output_routed_positive
```
