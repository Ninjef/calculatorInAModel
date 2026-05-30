# 2026-05-30 Additive Zero-Improvement Source Gate

## Question

Can zero-improvement result-boundary targets be made handoff-aware by scoring
forced-result utility through the non-bottleneck additive path instead of the
answer-decoder bottleneck?

This keeps the target answer-derived and does not force the true sum directly.
The intended mechanism was to make source policy acquisition care about the
same additive/readout geometry used by the trusted frozen-policy handoff.

## Code Change

Added:

```text
--result-boundary-target-mode additive_zero_improvement
```

This mode uses the same improvement-over-zero-injection target construction as
`zero_improvement`, but scores both forced result classes and the zero-injection
baseline with `calculator_bottleneck_mode="none"`.

Unit coverage verifies that the new target mode routes forced-result and
zero-injection scoring through the additive path.

## Run

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/source200_full_enum_cpu/2026-05-30_174833_075930_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Matched source-gate setup:

- `operand_max=19`
- full-grid `400` examples
- `200` source steps
- `answer_loss_weight=0`
- frozen product semantic decoder
- `result_boundary_target_loss_weight=1`
- `result_boundary_target_mode=additive_zero_improvement`

## Results

| Metric | Step 0 | Step 100 | Step 200 |
| --- | ---: | ---: | ---: |
| learned-best fraction | `0.0025` | `0.3875` | `0.6025` |
| hard-best equals true sum | `0.0325` | `0.0325` | `0.0325` |
| true-result target probability | `0.0237` | `0.0248` | `0.0225` |
| snapshot normal / calc | `0.0250` | `0.0350` | `0.0200` |
| injection-zero | `0.0575` | `0.0400` | `0.0475` |
| forced-random | `0.0175` | `0.0150` | `0.0150` |

Final eval:

```text
8/400 = 0.0200
```

The source learned the additive-path target (`learned_best=0.6025`) while the
target itself was non-arithmetic: the additive forced-result best class matched
the true sum on only `0.0325` of examples, and true-result target probability
was only `0.0225`.

## Decision

```text
additive_zero_improvement_naive_source_gate_negative
```

Naive additive-path zero-improvement is not a viable handoff-aware
source-shaping signal. The additive/readout path is untrained during source
acquisition, so its forced-result loss table assigns arbitrary result codes.

Do not run longer source or handoff jobs with this plain mode as novelty. If
continuing additive-path targets, first add a readout-preconditioning or
co-training mechanism and state how it avoids simply returning to prescriptive
true-result supervision.
