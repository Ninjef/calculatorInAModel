# Phase 7 Fifteenth Task: Online MLP Shadow-Feedback Validation Gate

## Objective

Test whether validation-selected checkpointing can rescue the simple online
MLP shadow-feedback warmup without tuning on the heldout test split.

## Method

- Add a validation split separate from the heldout test split.
- Select the shadow checkpoint by validation model-gradient agreement:
  `min(result-proj cosine, upstream cosine)`.
- Restore the selected shadow state and report final gate metrics on the
  untouched heldout test split.

## Result

```text
online_mlp_shadow_feedback_validation_selection_negative
```

The selected `h64` checkpoint at step `60` reached heldout-test
result/upstream cosines `0.6449/0.7266`, with train-test gaps
`0.3201/0.2414`. The final unselected checkpoint reached `0.6955/0.7617` on
heldout test, still below the result threshold.

## Decision

Do not launch Stage 1 from validation-selected simple online MLP shadow
feedback. Next work should change the shadow target/state or add stronger
regularization.
