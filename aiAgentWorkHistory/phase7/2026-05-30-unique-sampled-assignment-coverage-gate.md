# 2026-05-30 unique sampled assignment coverage gate

## Question

Was the uniform sampled hard-assignment failure mainly caused by duplicate
candidate waste, or does sparse result coverage remain insufficient even when
each prompt receives unique sampled candidates?

This is a coverage-aware follow-up to the sampled assignment cost gate. It
keeps the same scorer budget as sample16/sample32 but samples without
replacement per prompt.

## Implementation

Added:

- `--result-policy-improvement-assignment-unique-sampling`
- per-prompt without-replacement candidate sampling
- validation that unique sampling requires sampled assignment and cannot exceed
  the result vocabulary size

Candidates are still the learned result plus random classes, but duplicates
are removed by construction.

## Runs

All runs used the same op19 `rhead64`, product-decoder, one-negative
forced-margin source gate and CLI seed `41` / effective model seed `43`.

Existing comparators:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_exact_source200_cpu
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_sample16_source200_cpu
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_sample32_source200_cpu
```

Unique sampling runs:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_unique16_source200_cpu/2026-05-30_105130_228946_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_unique32_source200_cpu/2026-05-30_104934_907504_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed43
```

## Results

| Assignment | Scored results | Step-200 true coverage | Step-200 target acc | Best snapshot normal/calc | Final eval | Approx wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.9900` | `0.8625` at step `150` | `294/400 = 0.7350` | `115.5s` |
| Sample16 duplicate-prone | `16/39` | `0.6125` | `0.4581` | `0.3650` at step `200` | `141/400 = 0.3525` | `88.0s` |
| Unique16 | `16/39` | `0.6525` | `0.5317` | `0.3625` at step `200` | `162/400 = 0.4050` | `88.0s` |
| Sample32 duplicate-prone | `32/39` | `0.7400` | `0.6773` | `0.4050` at step `150` | `152/400 = 0.3800` | `105.5s` |
| Unique32 | `32/39` | `0.9275` | `0.8156` | `0.6250` at step `200` | `244/400 = 0.6100` | `100.5s` |

## Decision

```text
unique_sampled_assignment_coverage_mixed_positive_but_below_exact
```

Interpretation:

- Duplicate removal matters. Unique32 is much stronger than duplicate-prone
  sample32 at the same nominal scorer count.
- Sparse unique coverage is still not enough. Unique32 scores most of the
  39-class result vocabulary and reaches true coverage `0.9275`, but it still
  misses the exact ceiling by a wide margin.
- Unique16 remains too coverage-limited to be useful.

Do not run more unique-uniform count ladders on this op19 `rhead64` gate as
novelty. Future candidate-cost reduction needs a smarter non-uniform proposal,
active/uncertainty allocation, or different target construction that closes the
remaining gap to exact assignment at materially lower scorer cost.
