# 2026-05-30 Forced-Margin Product-Decoder Parity

## Question

Does the automated one-negative forced-margin recovery benchmark survive the
wider model when the semantic decoder uses the later product interaction?

This follows the scale-stress review direction. The prior wider stress used an
existing `n_embd=32`, `n_head=2` non-product semantic decoder. That was a valid
scale/stability result, but it left a product-decoder parity caveat.

## Scaffold

There was no existing wider product semantic-decoder checkpoint, so I first
trained a matching oracle decoder:

```text
runs/2026-05-30_phase7_product_decoder_parity/embd32_product_oracle_decoder_steps1000_cpu
```

Configuration:

- `n_embd=32`, `n_head=2`, `n_layer=2`
- `answer_decoder_interaction=product`
- `calculator_bottleneck_mode=answer_decoder`
- oracle operands, operand-spans readout

The oracle decoder reached final eval `1.0000`; oracle-at-eval snapshots were
`1.0000` by step `500`. Normal free-policy snapshots stayed low, as expected
for oracle decoder training, because this scaffold does not train a usable
result-space source policy.

## Runs

Product wider source:

```text
runs/2026-05-30_phase7_product_decoder_parity/embd32_product_forced_margin_source630_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_product_decoder_parity/embd32_product_handoff600_from_step630_cpu
```

The CLI seed was `24`; run directories record effective model seed `26`.

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Late recovery |
| ---: | ---: | ---: | ---: | ---: | --- |
| `570` | `0.6100` | `0.6100` | `0.0175` | `1.0000` | off |
| `600` | `0.6375` | `0.6375` | `0.0300` | `1.0000` | on |
| `630` | `0.9475` | `0.9475` | `0.0200` | `1.0000` | on |

Final source eval was `0.9475`. The source run recorded `23,791` parameters,
matching the wider non-product source scale.

Trusted product 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9725` | `0.0125` | `1.0000` | `0.9650` | `0.0175` |
| `200` | `0.9825` | `0.0175` | `0.9925` | `0.9425` | `0.0200` |
| `300` | `0.9950` | `0.0225` | `0.9900` | `0.9575` | `0.0275` |
| `400` | `0.9975` | `0.0050` | `0.9925` | `0.9575` | `0.0175` |
| `500` | `1.0000` | `0.0000` | `0.9875` | `0.9600` | `0.0250` |
| `600` | `1.0000` | `0.0000` | `1.0000` | `0.9700` | `0.0225` |

Final handoff eval was `1.0000`. The 128-sample diagnostic summary reported
normal `1.0000`, injection-zero `0.0000`, forced-random `0.0234`, and learned
calculator accuracy `0.9297`.

## Decision

```text
automated_forced_margin_recovery_wider_product_decoder_parity_positive
```

Interpretation:

- The wider forced-margin benchmark survives product-decoder parity. The prior
  wider non-product result was not an artifact of the older decoder
  interaction.
- The product decoder is at least as friendly to the staged recipe in this
  seed: source recovery rose from `0.6375` to `0.9475`, and handoff reached
  perfect normal accuracy with near-zero calculator-ablation controls.
- This strengthens the staged-transfer benchmark, but it still does not solve
  the final thesis. Source training still uses hard improvement assignment,
  true-result forced-margin pressure, a pretrained semantic decoder, and a
  frozen transferred policy.

Do not rerun this same `n_embd=32`, `n_head=2`, product-decoder, effective-seed
`26` source-plus-handoff as novelty. Further forced-margin work should move to
a new axis such as larger operand range, larger architecture family,
many-calculator cost, or removal of hard assignment / true-result forcing.
