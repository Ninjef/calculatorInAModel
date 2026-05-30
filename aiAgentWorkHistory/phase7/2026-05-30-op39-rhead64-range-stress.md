# 2026-05-30 op39 rhead64 Range Stress

## Question

Does the op29 hidden result-head capacity fix scale to a larger operand range,
or does exact full-grid staged source training become too costly or too weak?

This is a larger-range stress of the op29 `rhead64` recipe. It moves from
`operand_max=29` (`900` prompts, `59` result classes) to `operand_max=39`
(`1600` prompts, `79` result classes), using the same `n_embd=32`, `n_head=2`,
product answer decoder, one-negative forced-margin source objective, and
trusted frozen-policy additive handoff.

## Runs

Oracle decoder:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op39_product_oracle_decoder_steps1000_cpu
```

Interrupted source attempt:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op39_rhead64_source630_cpu
```

The full source attempt started at `09:25:25` and was stopped at `09:58:53`
after about `33` local CPU minutes because it had not completed, but it had
saved checkpoints through step `540`.

Step-540 eval and continuation:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op39_rhead64_step540_eval_cpu
runs/2026-05-30_phase7_forced_margin_range_stress/op39_rhead64_recovery_from_step540_cpu
```

Trusted frozen-policy additive handoff:

```text
runs/2026-05-30_phase7_forced_margin_range_stress/op39_rhead64_handoff600_from_recovery_step90_cpu
```

CLI seed was `37`; run directories record effective model seed `39`.

## Results

Oracle decoder:

| Step | Oracle snapshot |
| ---: | ---: |
| `100` | `0.5781` |
| `200` | `0.9300` |
| `300` | `1.0000` |
| `1000` | `1.0000` |

Final oracle eval was `1600/1600 = 1.0000`.

Source:

| Source stage | Normal / calc | Injection-zero | Oracle | Forced-random |
| --- | ---: | ---: | ---: | ---: |
| Step `540` eval | `0.5469` | `0.0181` | `1.0000` | `0.0075` |
| Continuation step `30` | `0.4513` | `0.0206` | `1.0000` | `0.0106` |
| Continuation step `60` | `0.5344` | `0.0200` | `1.0000` | `0.0156` |
| Continuation step `90` | `0.9431` | `0.0213` | `1.0000` | `0.0113` |

The step-540 zero-step eval final was `869/1600 = 0.5431`. The continuation
final eval was `1504/1600 = 0.9400`.

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.9456` | `0.0000` | `0.9981` | `0.9456` | `0.0144` |
| `200` | `0.9344` | `0.0000` | `0.9919` | `0.9331` | `0.0156` |
| `300` | `0.9450` | `0.0000` | `0.9888` | `0.9381` | `0.0138` |
| `400` | `0.9494` | `0.0000` | `0.9825` | `0.9388` | `0.0156` |
| `500` | `0.9375` | `0.0000` | `0.9819` | `0.9288` | `0.0113` |
| `600` | `0.9419` | `0.0000` | `0.9725` | `0.9375` | `0.0138` |

Final handoff eval was `1516/1600 = 0.9475`. The 128-sample diagnostic summary
reported normal `0.9844` and learned calculator accuracy `0.9609`, but the
full-grid snapshot above is the primary evidence.

## Decision

```text
op39_rhead64_range_stress_causal_but_costly_mixed_positive
```

Interpretation:

- The op39 result is causal: handoff normal accuracy is far above
  injection-zero and forced-random controls, and learned calculator accuracy is
  high.
- The result is not op29-style perfect. The source was only `0.543` at saved
  step `540`, recovered to `0.940` only after continuation, and handoff
  plateaued around `0.94-0.95`.
- The cost/scalability warning is now concrete. Exact full-grid source training
  at op39 was slow enough locally that the one-shot source run was stopped
  before step `630`. This supports prioritizing cheaper assignment,
  many-calculator cost accounting, or a materially different source-capacity /
  credit-assignment mechanism before op49-style full-grid scaling.

Do not rerun the same op39 effective-seed-39 full-grid `rhead64` source,
step-540 continuation, and 600-step handoff as novelty. Further full-grid range
tests need an explicit scalability hypothesis rather than just a larger grid.
