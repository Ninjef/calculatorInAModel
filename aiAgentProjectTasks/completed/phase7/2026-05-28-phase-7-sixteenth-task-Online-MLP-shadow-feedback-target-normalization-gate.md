# Phase 7 Sixteenth Task: Online MLP Shadow-Feedback Target-Normalization Gate

## Objective

Test whether fit-split target normalization makes the online MLP
shadow-feedback module generalize well enough to clear the heldout warmup gate.

## Method

- Fit per-result z-score statistics on the fit split only.
- Train the online MLP on normalized target gradients.
- Unnormalize predicted feedback before inducing model gradients.
- Select checkpoints with validation model-gradient agreement and report the
  final gate on an untouched heldout test split.

## Result

```text
online_mlp_shadow_feedback_target_normalization_partial_no_go
```

Target normalization improved heldout-test alignment. Best near miss was
hidden size `16`, with heldout-test result/upstream cosines `0.7259/0.7549`,
relative norms `1.4146/1.1848`, and train-heldout gaps `0.1723/0.1458`.

## Decision

Do not launch Stage 1 from this target-normalized sweep. Next work should
change shadow input/state or objective more substantially.
