# 2026-05-30 - Cached Teacher Target Table

## Question

Was frozen-teacher additive anchoring weak because the policy could not imitate
the target table, because repeated forced-result rescoring was costly/noisy, or
because the teacher table itself was not correct enough?

## Mechanism

Added `--result-boundary-target-cache`.

When enabled, the script builds the result-boundary target table once on the
exhaustive grid, optionally from `--result-boundary-target-teacher-checkpoint`.
Training then uses live result-policy logits against the cached table without
rescoring forced results every step.

Two cache modes were tested:

- `target_weights`: train against the cached zero-improvement distribution.
- `hard_best`: train CE to the cached teacher best result.

Teacher/student checkpoint:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/semantic_distill_precondition300_cpu/2026-05-30_175847_794736_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-addsemdist1-addsemdists8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

## Runs

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_teacher_anchor_target_weights_source800_cpu/2026-05-30_183104_370107_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rbtteacher-rbtcachetarget_weights-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_teacher_anchor_hard_best_source800_cpu/2026-05-30_183225_700272_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rbtteacher-rbtcachehard_best-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
runs/2026-05-30_phase7_additive_zero_improvement_boundary/cached_teacher_anchor_hard_best_source1600_cpu/2026-05-30_183411_831220_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rbtteacher-rbtcachehard_best-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
```

All runs used the 400-prompt full `0..19` grid and the semantic-distilled
additive teacher table.

## Results

| Gate | Step | best=true | learned-best | source calc | final eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| online frozen teacher | 800 | `0.5225` | `0.4125` | `0.1700` | `0.1750` |
| cached target weights | 800 | `0.5225` | `0.4000` | `0.1650` | `0.1650` |
| cached hard best | 800 | `0.5225` | `0.6675` | `0.3300` | `0.3375` |
| cached hard best | 1600 | `0.5225` | `0.7100` | `0.3575` | `0.3725` |

The 1600-step cached hard-best counterfactuals stayed causal but weak:
injection-zero `0.0391`, oracle-at-eval `1.0000`, forced-zero `0.0000`,
forced-random `0.0391`.

## Interpretation

Caching the soft target did not improve the online frozen-teacher result, so
the repeated forced-result scoring loop was not the main blocker. Hard-best
teacher imitation substantially improved policy uptake, showing that the soft
zero-improvement distribution was too weak/diffuse for this policy path.

However, the teacher's best result is true only `0.5225` of prompts. Better
imitation therefore improves final accuracy only to `0.3725`, not to a useful
source. This branch is now target-quality limited.

## Decision

Do not run more same-teacher cached soft/hard length, LR, or freezing sweeps as
novelty. Cached tables are useful as a cheap diagnostic for future target
constructions, but this teacher table is not good enough to be the training
method.
