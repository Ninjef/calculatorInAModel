# 2026-05-31 Online Hard Memory Semantic-Distill Routed Shared Output Op29 Stress

## Question

Does the four-hook shared-output online-hard-memory plus additive
semantic-distillation recipe still work at `operand_max=29`, or was the op19
routed/shared result a small-grid artifact?

This is a thesis-axis range/scaling check, not another op19 seed repeat. It
keeps the scalable shared output projection and shallow result head, while
using the existing op29 product semantic decoder.

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_op29_src800/2026-05-30_204714_735170_model-c-op0-29-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-483c97d99c/model-c-2digit-seed9
```

This first source accidentally used the script default `calculator_read_position=eq`.
It cleared source and handoff, but the matched op19/op29 recipe uses
`operand_spans`. The primary evidence below is therefore the matched rerun:

```text
runs/ohm_semdist_hooks4_shareout_op29_spans_src800/2026-05-30_210136_997305_model-c-op0-29-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-483c97d99c/model-c-2digit-seed9
```

Settings:

- CLI seed `7`, effective seed `9`.
- op29 exhaustive `900`-prompt grid, result vocab `59`.
- Wider product decoder geometry: `n_embd=32`, `n_head=2`, `n_layer=2`.
- Calculator read position `operand_spans`, span width `2`.
- Four calculator hooks with `left_operand_mod` routing.
- `--share-calculator-output-proj`.
- Shallow result head: `calculator_result_head_hidden_size=0`.
- Sparse zero-improvement online hard memory with topk8+unique24 candidates.
- Additive semantic distillation weight `1`, sample count `8`.
- Source bottleneck mode `answer_decoder`.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc | Distill agree | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0089 | 0.0244 | 0.0000 | 0.0178 | 0.0089 | 0.0708 | 21,600 |
| 200 | 1.0000 | 0.0278 | 0.0000 | 0.0133 | 1.0000 | 0.7040 | 367,200 |
| 400 | 1.0000 | 0.0167 | 0.0011 | 0.0144 | 1.0000 | 0.9870 | 367,200 |
| 600 | 1.0000 | 0.0178 | 0.0022 | 0.0156 | 1.0000 | 1.0000 | 367,200 |
| 800 | 1.0000 | 0.0233 | 0.0011 | 0.0200 | 1.0000 | 1.0000 | 367,200 |

Final source metrics:

- Final eval: `900/900 = 1.0000`.
- Diagnostic calculator-result accuracy: `1.0000`.
- Final 128-sample source counterfactuals: injection-zero `0.0078`,
  forced-zero `0.0000`, forced-random `0.0156`.
- Final routed diagnostic: all four routed hooks reached calculator-result
  accuracy `1.0000`.
- Online hard memory was full/frozen by step `50`; cumulative forced-result
  evaluations stayed at `367,200` from step `50` onward.

## Trusted Handoff

```text
runs/ohm_semdist_hooks4_shareout_op29_handoff600/2026-05-30_205452_274599_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

That first handoff used the `eq` source. The matched `operand_spans` handoff is
the primary trusted gate:

```text
runs/ohm_semdist_hooks4_shareout_op29_spans_handoff600/2026-05-30_210938_275842_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Settings:

- Loaded the source final checkpoint with `compatible_model`.
- Calculator read position `operand_spans`, span width `2`.
- Additive non-bottleneck mode: `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `--freeze-semantic-decoder`.
- `--freeze-calculator-policy`.
- Four routed hooks with shared output projection preserved.
- `600` downstream/readout steps.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0089 | 0.0000 | 0.0178 | 1.0000 |
| 100 | 1.0000 | 0.0000 | 0.0000 | 0.0133 | 1.0000 |
| 200 | 1.0000 | 0.0044 | 0.0000 | 0.0133 | 1.0000 |
| 300 | 1.0000 | 0.0211 | 0.0033 | 0.0122 | 1.0000 |
| 400 | 1.0000 | 0.0156 | 0.0011 | 0.0144 | 1.0000 |
| 500 | 1.0000 | 0.0156 | 0.0022 | 0.0067 | 1.0000 |
| 600 | 1.0000 | 0.0133 | 0.0022 | 0.0156 | 1.0000 |

Final handoff metrics:

- Final eval: `900/900 = 1.0000`.
- Final 128-sample diagnostic: normal `1.0000`, calculator-result accuracy
  `1.0000`; counterfactuals injection-zero `0.0156`, forced-zero `0.0000`,
  forced-random `0.0156`.
- Step-600 900-sample controls: injection-zero `0.0133`, forced-zero
  `0.0022`, forced-random `0.0156`.
- All four routed hooks had calculator-result accuracy `1.0000`.
- Active-only routing remained active: four hooks configured, one hook invoked
  per routed example batch partition.

## Interpretation

The routed/shared-output online-hard-memory plus semantic-distillation method
survives the op29 range stress with a shallow result head. This is stronger
than the earlier forced-margin op29 shallow-head stress, which missed and
needed `rhead64` to clear. Here the source and trusted non-bottleneck handoff
both clear at `1.0000`, and the controls show causal calculator use rather
than neuron-path bypass.

This does not solve the full thesis. The method still uses sparse forced-result
candidate scoring until the fixed-grid memory fills, then stores per-prompt
hard targets. The new result says fixed-grid routed/shared op19 and op29 are
not the next bottleneck. The next high-leverage test is streaming or
fresh-prompt memory, where per-prompt target memory cannot simply memorize the
entire grid.

## Decision

```text
online_hard_memory_semantic_distill_shared_output_routed_op29_range_positive
```
