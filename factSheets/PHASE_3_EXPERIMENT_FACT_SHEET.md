# Phase 3 Experiment Fact Sheet

## Track A: joint pair-action interface smoke

Date: 2026-05-06

### Setup

- Implemented `calculator_action_head=joint_pair` and `calculator_estimator=action_loss_full_enum_joint_interface`.
- Strict bottleneck was preserved: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, read position `operands`, `calculator_bottleneck_mode=answer_decoder`.
- Training targets used full-enum answer NLL over all `20 x 20` action pairs, converted to a soft pair distribution with temperature `1.0`.
- The joint objective trained pair logits directly and did not marginalize into independent A/B targets.
- Primary claim constraints held: `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_upstream_encoder=true`, and trainable parameters were limited to `calculator_hook.pair_proj`.

### Smoke Run

Run:

```text
runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103
```

Start checkpoint:

```text
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
```

Result summary:

| Metric | Value |
| --- | ---: |
| Final built-in eval exact | 0.04883 |
| Snapshot step-200 normal exact | 0.06250 |
| Snapshot step-200 injection-zero exact | 0.00000 |
| Snapshot step-200 oracle-at-eval exact | 0.92188 |
| Snapshot step-200 pair exact | 0.00000 |
| Snapshot step-200 calc result acc | 0.05469 |
| Snapshot step-200 mean pair entropy | 5.99086 |

### Post-Smoke Diagnostics

Canonical causal:

| Metric | Value |
| --- | ---: |
| Normal exact | 0.04688 |
| Injection-zero exact | 0.00000 |
| Forced-random exact | 0.06250 |
| Oracle-at-eval exact | 0.90625 |
| Pair exact | 0.00000 |
| Result-equivalent pair accuracy | 0.04688 |
| Classification | `calculator_ignored_or_bypassed` |
| Bottleneck label | `strict_bottleneck_unvalidated` |

Full-enum joint diagnostic:

| Metric | Value |
| --- | ---: |
| Best NLL | 0.09967 |
| Learned NLL | 5.72066 |
| True NLL | 0.09967 |
| Learned-minus-true gap | 5.62099 |
| Learned-minus-best gap | 5.62100 |
| Learned-best fraction | 0.00000 |
| Tie-aware learned-best <= 1e-3 | 0.10938 |
| Best result matches true sum | 0.90625 |
| Learned result matches true sum | 0.03125 |
| Teacher effective pairs | 29.21159 |
| Pair-logit effective pairs | 399.75911 |

Private protocol:

| Metric | Value |
| --- | ---: |
| All-pair answer exact | 0.04500 |
| Operand exact | 0.00000 |
| Pair exact | 0.00000 |
| Calculator result accuracy | 0.02750 |
| Learned result distribution | `{"12": 316, "21": 84}` |

### Finding

Useful negative smoke result. The full-enum teacher remains informative: true actions have very low NLL and the true sum is best/result-best often. But the direct joint pair head did not learn the teacher distribution in 200 steps from the selected checkpoint. Its output distribution stayed nearly uniform by entropy while argmax actions collapsed to a small set of result classes. Because pair exact, result-equivalent pair accuracy, private-protocol accuracy, and learned-best fraction did not improve, the selected-checkpoint ladder and Stage-B ladder were not run.

### Recommendation

No-go on continuing this exact Track A implementation to the primary ladder. Move next to Track B: an identifiability curriculum where task structure rewards operand identity directly, so addition-only result equivalence cannot hide incorrect pairs.
