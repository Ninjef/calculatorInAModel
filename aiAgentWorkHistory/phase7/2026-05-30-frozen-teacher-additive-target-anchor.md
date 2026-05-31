# 2026-05-30 - Frozen-Teacher Additive Target Anchor

## Question

Can the semantic-distilled additive readout table be frozen as an
answer-derived teacher so the live source policy learns useful result choices
without the target drifting?

## Mechanism

Added `--result-boundary-target-teacher-checkpoint`. When set,
`result_boundary_target_loss` computes the forced-result loss table and
zero-injection baseline with a separate frozen teacher model, while the live
model still supplies the trained result-policy logits. This tests target
anchoring separately from policy uptake.

Also added `--freeze-post-calculator-decoder` as a diagnostic freezer for the
post-hook transformer blocks and final LM readout.

Teacher/student checkpoint:

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/semantic_distill_precondition300_cpu/2026-05-30_175847_794736_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-addsemdist1-addsemdists8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4/final_weights.pt
```

## Runs

```text
runs/2026-05-30_phase7_additive_zero_improvement_boundary/frozen_table_policy_only_source200_cpu/2026-05-30_181001_317743_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
runs/2026-05-30_phase7_additive_zero_improvement_boundary/frozen_additive_scorer_policy_source200_cpu/2026-05-30_181220_176750_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-freezepostcalc-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
runs/2026-05-30_phase7_additive_zero_improvement_boundary/frozen_teacher_anchor_source200_cpu/2026-05-30_181630_211887_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rbtteacher-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
runs/2026-05-30_phase7_additive_zero_improvement_boundary/frozen_teacher_anchor_source800_cpu/2026-05-30_181740_705293_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-additive_zero_improvement-rbtt1-rbtchunk64-rbtteacher-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed6
```

All runs used the 400-prompt full `0..19` grid, full 39-result enumeration,
`result_boundary_target_mode=additive_zero_improvement`, and the semantic
preconditioner above.

## Results

| Gate | Step | best=true | learned-best | source calc | final eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| freeze whole encoder/readout | 200 | `0.5225` | `0.1880` | `0.0275` | `0.0225` |
| freeze post-calculator decoder | 200 | `0.1575` | `0.6575` | `0.0850` | `0.0900` |
| frozen teacher anchor | 200 | `0.5225` | `0.3325` | `0.1025` | `0.0975` |
| frozen teacher anchor | 800 | `0.5225` | `0.4125` | `0.1700` | `0.1750` |

The 800-step teacher-anchor run had final counterfactuals: injection-zero
`0.0391`, oracle-at-eval `1.0000`, forced-zero `0.0000`, forced-random
`0.0391`.

## Interpretation

Frozen target construction fixes target drift, but it does not make the source
policy learn the repaired target well enough. The policy can move toward the
teacher table (`learned_best` rises from `0.0025` to `0.4125`), but true
calculator-result accuracy remains weak (`0.1700`) and final eval is only
`0.1750`.

This is a mixed-negative for additive-path target anchoring. It is useful as a
diagnostic and possibly as a component, but not as the training method.

## Decision

Do not run more same-checkpoint frozen-teacher anchor length/LR/freezing sweeps
as novelty. The next mechanism should change policy uptake directly: easier
policy target parameterization, direct source-logit training against cached
teacher tables, or a different estimator that preserves target quality while
raising true-result uptake.
