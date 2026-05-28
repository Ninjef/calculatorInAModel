# Phase 7 Twentieth Task: Online MLP Shadow-Feedback Gap-Penalized Selection Gate

## Objective

Test whether selecting directional-loss shadow checkpoints with an explicit
train-validation gap penalty clears the heldout warmup gate.

## Method

- Add `gap_penalized_min_cosine` checkpoint selection.
- Score validation checkpoints as validation min-cosine minus a weighted
  train-validation cosine gap.
- Keep the heldout test split untouched for final gate reporting.
- Test the directional-loss `injection_grad_logits` setup that previously
  passed heldout cosine but missed the gap gate.

## Result

```text
online_mlp_shadow_feedback_gap_penalized_selection_tradeoff_no_go
```

Gap-penalized selection exposed a smooth tradeoff but did not clear both gates.
For `cosine` h16, penalty `4.0` selected step `70` and kept heldout
result/upstream cosines above threshold (`0.7165/0.7439`), but result gap
remained `0.1673`. Penalty `5.0` selected step `60`, reducing gaps to
`0.1511/0.1220`, but heldout fell below threshold (`0.6872/0.6979`).

## Decision

Do not launch Stage 1 from checkpoint selection alone. Next work should use
training-time regularization, target stabilization, or a different
learned-gradient state.
