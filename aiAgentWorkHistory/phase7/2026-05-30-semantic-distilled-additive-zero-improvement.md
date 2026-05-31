# 2026-05-30 Semantic-Distilled Additive Zero-Improvement

## Question

Can additive-path zero-improvement be rescued by first teaching the additive
non-bottleneck readout what arbitrary calculator result classes mean?

The prior naive additive-path target failed because the additive forced-result
loss table was non-arithmetic before the additive readout was trained. This
test adds readout preconditioning that does not tell the source policy which
result to request for a prompt.

## Mechanism

Added:

```text
--additive-semantic-distill-weight
--additive-semantic-distill-sample-count
--additive-semantic-distill-temperature
```

For sampled arbitrary forced result classes, the auxiliary compares:

- teacher: `calculator_bottleneck_mode=answer_decoder`;
- student: `calculator_bottleneck_mode=none`.

The loss is masked KL from additive-path logits to frozen semantic
answer-decoder logits. It teaches result readout semantics but does not use the
prompt's true sum as a target.

## Runs

Co-training:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/semantic_distill_source200_full_enum_cpu/2026-05-30_175724_134625_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-addsemdist1-addsemdists4-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Distill-only preconditioner:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/semantic_distill_precondition300_cpu/2026-05-30_175847_794736_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-addsemdist1-addsemdists8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Preconditioned source with ongoing distill:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/preconditioned_source200_full_enum_cpu/2026-05-30_175956_486599_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-addsemdist1-addsemdists8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Preconditioned source with distill off:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/preconditioned_source200_no_distill_cpu/2026-05-30_180123_978018_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

## Results

| Gate | Target quality | Policy uptake | Calc / normal |
| --- | ---: | ---: | ---: |
| naive additive zero-improvement | best=true `0.0325`, true prob `0.0225` | learned-best `0.6025` | `0.0200` |
| co-train distill + source | best=true `0.1775`, true prob `0.0991` | learned-best `0.7100` | snapshot `0.0825`, eval `0.0450` |
| distill precondition only | token agreement `0.7694` | n/a | n/a |
| preconditioned source + distill | best=true `0.8200`, true prob `0.2863` | learned-best `0.1400` | snapshot `0.0675`, eval `0.0830` |
| preconditioned source, distill off | best=true `0.1575`, true prob `0.1035` | learned-best `0.6950` | snapshot `0.0900`, eval `0.0620` |

## Interpretation

Semantic distillation fixes the first failure mode. After preconditioning, the
additive target table is much more arithmetic: best=true is already `0.5225`
at source step `0` and reaches `0.8200` with ongoing distillation.

But the source policy does not learn the repaired target strongly. With ongoing
distillation, learned-best is only `0.1400` at step `200`. Without ongoing
distillation, the table drifts back toward non-arithmetic preferences and the
policy learns that drifting wrong target (`learned_best=0.6950`, best=true
`0.1575`).

## Decision

```text
semantic_distilled_additive_zero_improvement_mixed_negative
```

This is useful mechanistic evidence, but not a source-learning success.
Readout semantics are a necessary piece for additive-path target construction,
yet they are not sufficient. Do not run more plain distill weight/sample/length
tweaks as novelty.

Next useful work needs to explicitly solve policy uptake or target drift, for
example by freezing/protecting the repaired additive readout table during
source updates, anchoring the policy target to a repaired table, or using a
different estimator that can increase true-result uptake once the table is
meaningful.
