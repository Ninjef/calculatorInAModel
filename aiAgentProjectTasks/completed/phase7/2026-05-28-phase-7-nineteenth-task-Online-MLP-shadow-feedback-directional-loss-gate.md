# Phase 7 Nineteenth Task: Online MLP Shadow-Feedback Directional-Loss Gate

## Objective

Test whether training the online MLP shadow module with a directional loss
clears the heldout warmup gate.

## Method

- Add `cosine` loss for normalized-target direction.
- Add `mse_plus_cosine` for combined MSE and directional fitting.
- Keep `mse` as the default.
- Use the target-normalized simple `injection_grad_logits` state, since that
  was the prior near-miss before raw policy features and feature z-scoring.
- Gate h8/h16/h32 with validation checkpoint selection and heldout-test
  model-gradient agreement.

## Result

```text
online_mlp_shadow_feedback_directional_loss_partial_no_go
```

Directional losses materially improved heldout cosines. `cosine` h16/h32
reached heldout result/upstream cosines `0.7646/0.8007` and `0.7937/0.8270`;
`mse_plus_cosine` h16/h32 reached `0.7785/0.8112` and `0.7853/0.8174`.

The full gate still failed because result train-heldout gaps stayed around
`0.20`, above the `0.15` gap fence. Smaller h8 missed the heldout cosine
threshold.

## Decision

Do not launch Stage 1 from plain directional loss. Next work should add
explicit norm/gap regularization, use a more stable target construction, or
change the learned-gradient state more substantially.
