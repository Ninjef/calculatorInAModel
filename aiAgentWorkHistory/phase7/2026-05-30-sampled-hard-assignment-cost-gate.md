# 2026-05-30 sampled hard-assignment Cost Gate

## Question

Can we reduce the cost of hard improvement assignment by scoring only a sampled
subset of result classes per prompt while preserving the source-policy signal
that makes the staged `rhead64` forced-margin recipe work?

Scalability hypothesis: if full result-class scoring is the compute bottleneck,
then scoring the learned result plus `16` or `32` uniform candidates should keep
a meaningful fraction of the exact source lift while reducing assignment
compute. The control is an exact full-result assignment ceiling with the same
op19 `rhead64`, product-decoder, one-negative forced-margin source recipe.

## Implementation

Added:

- `--result-policy-improvement-assignment-sample-count`
- sampled candidate construction from learned result plus uniform random result
  classes
- candidate-only forced-result scoring
- candidate-only hard improvement assignment with coverage diagnostics

Logged diagnostics include scored candidate count, true-result coverage, and
unique candidate fraction so failures can be separated into weak scoring
coverage versus downstream wiring.

## Runs

All runs used the same product semantic decoder:

```text
runs/2026-05-30_phase7_product_decoder_parity/embd32_product_oracle_decoder_steps1000_cpu/2026-05-29_210422_659184_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed26/checkpoint_snapshots/step_01000_weights.pt
```

Exact ceiling:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_exact_source200_cpu/2026-05-30_102233_252420_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

Sample16:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_sample16_source200_cpu/2026-05-30_102444_279398_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

Sample32:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_sample32_source200_cpu/2026-05-30_102646_576099_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

CLI seed was `41`; run directories record effective model seed `43`.

## Results

| Assignment | Scored results | Step-200 true coverage | Step-200 target acc | Best snapshot normal | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.9900` | `0.8625` at step `150` | `294/400 = 0.7350` |
| Sample16 | `16/39` | `0.6125` | `0.4581` | `0.3650` at step `200` | `141/400 = 0.3525` |
| Sample32 | `32/39` | `0.7400` | `0.6773` | `0.4050` at step `150` | `152/400 = 0.3800` |

Approximate local wall time from run start to metrics mtime:

| Assignment | Approx wall time |
| --- | ---: |
| Exact | `115s` |
| Sample16 | `88s` |
| Sample32 | `106s` |

Controls stayed low in the sampled branches, so the issue was not bypass or
decoder wiring. The sampled assignment simply did not provide a reliable target
stream. Duplicate candidates lowered unique coverage, and true-result coverage
remained too low even at `32/39` scored classes.

## Decision

```text
uniform_sampled_hard_assignment_disproven_for_op19_rhead64_cost_gate
```

Interpretation:

- Uniform sparse candidate scoring is not a viable cheap replacement for exact
  hard assignment in the current source recipe.
- The result is a direct assignment-cost test, not another local-target proposal
  variant, but it echoes the same lesson: sparse uniform coverage is not enough
  when the target depends on finding the right result class.
- The wall-clock win in this local gate is too small to justify the source
  degradation, especially because snapshots and other objectives still dominate
  part of the runtime.

Do not run more uniform sample-count ladders on this op19 `rhead64` 200-step
gate as novelty. Further assignment-cost work should use coverage-aware,
active, structured, or non-enumerative credit assignment and compare against an
exact assignment ceiling.
