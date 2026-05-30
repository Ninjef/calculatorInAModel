# 2026-05-30 Forced-Margin op29 Range Stress

## Question

Does the automated one-negative forced-margin recovery benchmark survive a
larger operand range?

This is a range-scaling stress, not a local forced-margin knob sweep. The
previous product-decoder parity result cleared `operand_max=19` with a wider
`n_embd=32`, `n_head=2`, product semantic decoder. This run keeps that wider
product setup but increases the full training grid from `20 x 20 = 400`
prompts to `30 x 30 = 900` prompts and increases the result vocabulary from
`39` to `59` classes.

## Scaffold

There was no existing wider product semantic-decoder checkpoint for
`operand_max=29`, so I first trained a matching oracle decoder:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_product_oracle_decoder_steps1000_cpu
```

Configuration:

- `operand_max=29`, `calculator_operand_vocab_size=30`
- `n_embd=32`, `n_head=2`, `n_layer=2`
- `answer_decoder_interaction=product`
- `calculator_bottleneck_mode=answer_decoder`
- oracle operands, operand-spans readout

The oracle decoder reached full-grid final eval `900/900 = 1.0000`; oracle
snapshots were `1.0000` by step `500`.

## Runs

Product op29 source:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_product_forced_margin_source630_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op29_product_handoff600_from_step630_cpu
```

The CLI seed was `27`; run directories record effective model seed `29`.

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Late recovery |
| ---: | ---: | ---: | ---: | ---: | --- |
| `570` | `0.4411` | `0.4411` | `0.0278` | `1.0000` | off |
| `600` | `0.3533` | `0.3533` | `0.0233` | `1.0000` | on |
| `630` | `0.6889` | `0.6889` | `0.0233` | `1.0000` | on |

Final source eval was `642/900 = 0.7133`. Late recovery still helped, but the
source remained far below the op19 product-parity source (`0.9475` final).

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.7144` | `0.0511` | `0.9556` | `0.6989` | `0.0144` |
| `200` | `0.7467` | `0.0389` | `0.8667` | `0.6878` | `0.0156` |
| `300` | `0.7944` | `0.0389` | `0.8656` | `0.6744` | `0.0200` |
| `400` | `0.8222` | `0.0433` | `0.8511` | `0.6811` | `0.0156` |
| `500` | `0.8189` | `0.0433` | `0.8689` | `0.6933` | `0.0211` |
| `600` | `0.8278` | `0.0344` | `0.8411` | `0.6522` | `0.0189` |

Final handoff eval was `768/900 = 0.8533`. The 128-sample diagnostic summary
reported normal `0.8281`, injection-zero `0.0313`, forced-random `0.0313`,
and learned calculator accuracy `0.6875`.

## Decision

```text
automated_forced_margin_recovery_op29_range_stress_mixed_negative
```

Interpretation:

- The staged recipe remains calculator-causal at op29: zero-injection and
  forced-random controls stay low while normal accuracy is far above controls.
- It does not clear the high non-bottleneck gate at this larger range. Source
  acquisition is the obvious bottleneck: even after late recovery, source calc
  was only `0.6889` on the full-grid snapshot and `0.7133` final eval.
- This is evidence against simply scaling the current full-grid hard-assignment
  forced-margin recipe to larger ranges. The recipe still works as an op19
  benchmark, but range scaling needs a changed source objective, a better
  scalable assignment approximation, or materially more efficient training.

Do not rerun this same `operand_max=29`, product-decoder, effective-seed `29`
source-plus-handoff as novelty. Do not jump straight to op49 with the same
full-grid hard-assignment recipe unless the goal is explicitly to measure
compute cost rather than improve the method. Further range work should change
the source objective, reduce assignment cost with a predeclared ceiling
comparison, or test whether additional source capacity/recovery changes the
range failure mode.
