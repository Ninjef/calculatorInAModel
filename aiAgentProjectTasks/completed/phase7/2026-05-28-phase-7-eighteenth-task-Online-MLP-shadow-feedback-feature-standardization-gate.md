# Phase 7 Eighteenth Task: Online MLP Shadow-Feedback Feature-Standardization Gate

## Objective

Test whether fit-split feature standardization rescues the target-normalized
online MLP shadow-feedback gate.

## Method

- Add `fit_zscore_per_feature` for online shadow input features.
- Fit feature mean/std on the fit split only.
- Apply the same feature transform to train, validation, and heldout examples
  before the shadow MLP.
- Keep target normalization, validation checkpoint selection, and raw-space
  model-gradient diagnostics unchanged.
- Test both `injection_grad_policy_state` and the simpler
  `injection_grad_logits` feature states at hidden sizes `16` and `32`.

## Result

```text
online_mlp_shadow_feedback_feature_standardization_negative
```

Feature standardization did not clear the heldout gate. Policy-state h16/h32
reached heldout result/upstream cosines `0.5942/0.3997` and `0.4340/0.4023`.
The simpler logits-state h32 reached `0.6691/0.7028`, but gaps were
`0.2830/0.2658`; logits-state h16 had a small result gap but missed upstream
with `0.6436/0.4763`.

## Decision

Do not launch Stage 1 from plain feature z-scoring. Next work should change
objective, regularization, or target construction rather than rerunning feature
scale changes alone.
