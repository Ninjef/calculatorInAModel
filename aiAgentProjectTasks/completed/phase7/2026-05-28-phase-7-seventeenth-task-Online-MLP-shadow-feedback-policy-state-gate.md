# Phase 7 Seventeenth Task: Online MLP Shadow-Feedback Policy-State Feature Gate

## Objective

Test whether richer result policy-state features rescue the target-normalized
online MLP shadow-feedback gate.

## Method

- Add a selectable shadow-feature mode.
- Keep the old `injection_grad_logits` mode as the default.
- Add `injection_grad_policy_state`, which appends result probabilities,
  log-probabilities, and entropy to the answer-gradient plus result-logit
  input state.
- Keep fit-split per-result target normalization, validation checkpoint
  selection, and heldout-test model-gradient comparison unchanged.

## Result

```text
online_mlp_shadow_feedback_policy_state_raw_features_negative
```

Raw policy-state features did not clear the heldout gate. Hidden size `32`
reached heldout-test result/upstream cosines `0.7037/0.7611`, but
train-heldout gaps were `0.2853/0.2131`. Hidden size `16` missed the result
threshold with heldout-test result/upstream cosines `0.6862/0.7391`.

The raw log-probability block dominated feature scale (`382.84` fit L2 versus
`69.50` input-gradient L2), making simple appending a poor state change.

## Decision

Do not launch Stage 1 from raw policy-state features. Next work should change
feature scaling/standardization, regularization, loss shape, or target
construction rather than rerunning this raw feature mode.
