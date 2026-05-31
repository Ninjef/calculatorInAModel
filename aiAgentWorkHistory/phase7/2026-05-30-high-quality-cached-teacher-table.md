# 2026-05-30 High-Quality Cached Teacher Table

## Question

The previous cached additive teacher-table gate showed that cached hard-best is
much easier for the source policy to imitate than cached soft
zero-improvement weights, but the teacher table itself was only true for
`0.5225` of prompts. Does a higher-quality additive teacher table turn cached
hard-best imitation into a strong source, or does the uptake/target-quality gap
remain?

## Mechanism

Reuse the preconditioned + ongoing semantic-distill additive source checkpoint
as the frozen teacher/cache source. This checkpoint previously showed much
better additive target quality (`best_true=0.8200` at source step 200) than the
distill-only preconditioner used in the first cached-teacher test.

Run the same source-policy training setup in cached target mode:

- `target_weights` cache, 800 steps.
- `hard_best` cache, 800 steps.
- `hard_best` cache, 1600 steps.

This is intentionally a diagnostic, not a scalable recipe: the cache still
comes from full-enum forced-result scoring with a teacher checkpoint.

## Runs

Teacher checkpoint:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/preconditioned_source200_full_enum_cpu/2026-05-30_175956_486599_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-addsemdist1-addsemdists8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

Run roots:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_high_quality_teacher_target_weights_source800_cpu
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_high_quality_teacher_hard_best_source800_cpu
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_high_quality_teacher_hard_best_source1600_cpu
```

## Results

| Gate | Teacher quality | Source uptake | Calc/final |
| --- | ---: | ---: | ---: |
| distill-only cached hard best, 1600 | best=true `0.5225` | learned-best `0.7100` | `0.3575` / `0.3725` |
| high-quality cached weights, 800 | best=true `0.8200` | learned-best `0.3925` | `0.2975` / `0.2725` |
| high-quality cached hard best, 800 | best=true `0.8200` | learned-best `0.7275` | `0.5950` / `0.5625` |
| high-quality cached hard best, 1600 | best=true `0.8200` | learned-best `0.7650` | `0.6175` / `0.5825` |

The 1600-step hard-best curve improved through about step 1250
(`learned_best=0.785`, calc around `0.645`) and then oscillated slightly by
final evaluation.

## Interpretation

Better target quality matters. Moving from a `0.5225` true teacher-best table
to a `0.8200` true teacher-best table raised final accuracy from `0.3725` to
`0.5825` under the same cached hard-best imitation mechanism.

Hardening the teacher table matters too. Soft cached zero-improvement weights
from the same high-quality teacher stayed weak (`0.2725` final), while
hard-best imitation produced meaningful source lift.

This still does not solve the project goal. The method remains cached,
teacher-dependent, and full-enum; it is also far below the teacher table's
`0.8200` best-true ceiling and below mature bottleneck zero-improvement source
quality. The result is a useful ceiling/diagnostic for target construction, not
a scalable training method.

## Decision

Do not run more same-teacher cached hard-best length/LR sweeps as novelty. Use
cached hard-best only as a cheap diagnostic for genuinely better
answer-derived target construction, or move to a mechanism that can generate
high-quality hard targets online/scalably without true-result forcing.
