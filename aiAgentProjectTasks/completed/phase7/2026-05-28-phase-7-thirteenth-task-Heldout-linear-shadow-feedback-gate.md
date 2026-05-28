# Phase 7 Thirteenth Task: Heldout Linear Shadow-Feedback Gate

## Mission

Check whether the fit-once linear shadow-feedback Stage 0 alignment
generalizes off its calibration examples.

## Claim Tested

Does a linear map from answer-loss injection gradients to boundary
result-logit gradients still induce boundary-aligned model updates on heldout
natural `0..19` exact-grid prompts?

## Decision

```text
heldout_linear_shadow_feedback_stage0_generalization_negative
```

The same-batch gate was too optimistic. Train result-proj cosine stayed near
perfect, but heldout result-proj cosine fell below the online-shadow warmup
threshold and the train-heldout gap was large.

## Stop Rule

Do not use same-batch linear shadow alignment as a training-budget gate. The
next branch should be an online MLP shadow module with result-policy state,
heldout warmup validation, and only then a 200-step early-lift smoke.
