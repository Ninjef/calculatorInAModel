# 2026-05-30 Forced-Margin Wider-Model Scale Stress

## Question

Does the automated one-negative forced-margin recovery benchmark survive a
wider model?

This is a scale/stability stress, not forced-margin knob tuning. The source
uses the same one-negative forced-margin recovery schedule as the tiny
benchmark, but with an existing wider semantic-decoder checkpoint:

- `n_embd=32`
- `n_head=2`
- `n_layer=2`
- operand-spans readout

Caveat: the available wider semantic decoder is the older non-product
`answer_decoder_interaction=none` checkpoint, not the later product decoder.
So this tests scaling of the staged forced-margin recipe under a wider model,
not exact parity with the product-decoder recipe.

## Setup Smoke

Run root:

```text
runs/2026-05-30_phase7_forced_margin_scale_stress/embd32_source_smoke
```

The zero-step smoke loaded the wider semantic decoder with
`semantic_decoder_only`, initialized a result-space policy, and confirmed the
oracle path remained healthy (`oracle=0.9300` on the 400-sample snapshot).

## Runs

Fresh wider source:

```text
runs/2026-05-30_phase7_forced_margin_scale_stress/embd32_source630_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_forced_margin_scale_stress/embd32_handoff600_from_step630_cpu
```

The CLI seed was `23`; the run directory records effective model seed `25`.

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Late recovery |
| ---: | ---: | ---: | ---: | ---: | --- |
| `570` | `0.8030` | `0.8625` | `0.0150` | `0.9600` | off |
| `600` | `0.7830` | `0.8450` | `0.0075` | `0.9475` | on |
| `630` | `0.8825` | `0.9150` | `0.0075` | `0.9350` | on |

Final source eval was `0.9125`. The wider source has `23,791` parameters in
the recorded metrics, versus `8,863` in the tiny forced-margin source run.

Trusted wider 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9675` | `0.0250` | `0.9850` | `0.8725` | `0.0475` |
| `200` | `1.0000` | `0.0375` | `1.0000` | `0.8750` | `0.0300` |
| `300` | `1.0000` | `0.0425` | `1.0000` | `0.8850` | `0.0300` |
| `400` | `1.0000` | `0.0550` | `1.0000` | `0.9175` | `0.0225` |
| `500` | `1.0000` | `0.0600` | `1.0000` | `0.8800` | `0.0375` |
| `600` | `1.0000` | `0.0625` | `1.0000` | `0.8850` | `0.0325` |

Final handoff eval was `1.0000`. The 128-sample diagnostic summary reported
normal `1.0000` and learned calculator accuracy `0.8906`.

## Decision

```text
automated_forced_margin_recovery_wider_model_scale_positive
```

Interpretation:

- The automated forced-margin recovery benchmark survives this wider
  architecture and handoffs extremely strongly.
- The handoff remains calculator-causal: zero-injection and forced-random
  controls stay low while normal/oracle are perfect by step `200+`.
- The wider model is not an answer to the final project goal because training
  remains prescriptive: source training still uses hard improvement assignment,
  true-result forced-margin pressure, a frozen transferred policy, and a
  pre-trained semantic decoder.
- The non-product decoder caveat matters. Future scale claims should either
  train a matching product semantic decoder or explicitly compare product and
  non-product source/handoff behavior.

Do not rerun this same `n_embd=32`, `n_head=2`, effective-seed-25 wider
source-plus-handoff as novelty. Further scale work should move to a larger
operand range, a matching product decoder, a larger architecture family, or
less-prescriptive source objectives.
