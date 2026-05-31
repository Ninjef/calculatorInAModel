# 2026-05-31 Online Hard Memory Semantic-Distill Routed Shared Output Fresh Seed

## Question

Does the four-hook shared-output online-hard-memory plus additive
semantic-distillation result replicate on a fresh seed?

This is the immediate robustness check for the routed/shared result. The
single-hook fresh semantic-distilled source reached perfect calculator accuracy
but missed trusted handoff, so this run uses CLI seed `7` / effective seed `9`,
matching that handoff-sensitive lineage.

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_fresh_src800/2026-05-30_203344_153653_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-c00fadadaf/model-c-2digit-seed9
```

Settings:

- CLI seed `7`, effective seed `9`.
- op19 exhaustive `400`-prompt grid.
- Four calculator hooks with `left_operand_mod` routing.
- `--share-calculator-output-proj`.
- Sparse zero-improvement online hard memory with topk8+unique24 candidates.
- Additive semantic distillation weight `1`, sample count `8`.
- Source bottleneck mode `answer_decoder`.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc | Distill agree | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0425 | 0.0275 | 0.0000 | 0.0250 | 0.0425 | 0.0340 | active |
| 200 | 0.7825 | 0.0400 | 0.0025 | 0.0125 | 0.7825 | 0.4610 | 86,400 |
| 400 | 1.0000 | 0.0475 | 0.0025 | 0.0150 | 1.0000 | 0.6990 | 86,400 |
| 600 | 1.0000 | 0.0450 | 0.0075 | 0.0125 | 1.0000 | 0.7386 | 86,400 |
| 800 | 1.0000 | 0.0575 | 0.0100 | 0.0275 | 1.0000 | 0.8005 | 86,400 |

Final source metrics:

- Final eval: `400/400 = 1.0000`.
- Diagnostic calculator-result accuracy: `1.0000`.
- Final 128-sample source counterfactuals: injection-zero `0.0703`,
  forced-zero `0.0078`, forced-random `0.0156`.
- Final routed diagnostic: all four routed hooks reached calculator-result
  accuracy `1.0000`.

## Trusted Handoff

```text
runs/ohm_semdist_hooks4_shareout_fresh_handoff600/2026-05-30_203641_075497_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
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
| 0 | 0.3150 | 0.0425 | 0.0000 | 0.0175 | 1.0000 |
| 100 | 1.0000 | 0.0475 | 0.0000 | 0.0150 | 1.0000 |
| 200 | 1.0000 | 0.0450 | 0.0025 | 0.0125 | 1.0000 |
| 300 | 1.0000 | 0.0525 | 0.0025 | 0.0275 | 1.0000 |
| 400 | 1.0000 | 0.0450 | 0.0025 | 0.0175 | 1.0000 |
| 500 | 1.0000 | 0.0600 | 0.0050 | 0.0325 | 1.0000 |
| 600 | 1.0000 | 0.0525 | 0.0075 | 0.0125 | 1.0000 |

Final handoff metrics:

- Final eval: `400/400 = 1.0000`.
- Final 128-sample counterfactuals: injection-zero `0.1094`,
  forced-zero `0.0078`, forced-random `0.0156`.
- Diagnostic calculator-result accuracy: `1.0000`.
- All four routed hooks had calculator-result accuracy `1.0000` in the final
  routed diagnostic summary.

## Interpretation

The routed/shared-output result replicated on the fresh seed. This is stronger
than the single-hook fresh semantic-distill result: the same seed lineage that
missed single-hook trusted handoff (`0.6475` final / `0.6625` step-600 normal)
clears the four-hook shared-output handoff perfectly.

The result suggests that routed/shared output plus semantic distillation can
make the additive readout geometry easier, not merely preserve a lucky original
source. It still does not complete the thesis because the run remains fixed
op19, prompt-memory based, and dependent on sparse forced-result scoring before
the memory fills. More op19 routed/shared seed repeats are now lower value than
streaming/fresh-prompt memory or larger-range routed/shared stress.

## Decision

```text
online_hard_memory_semantic_distill_shared_output_routed_fresh_seed_positive
```
