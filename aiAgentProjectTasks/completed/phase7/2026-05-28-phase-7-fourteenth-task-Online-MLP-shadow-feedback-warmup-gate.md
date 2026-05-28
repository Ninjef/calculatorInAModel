# Phase 7 Fourteenth Task: Online MLP Shadow-Feedback Warmup Gate

## Objective

Test whether an online MLP shadow-feedback module can generalize the boundary
result-logit gradient from a fit split to heldout examples before any Stage 1
training budget is spent.

## Method

- Add an `online_mlp` shadow-feedback diagnostic mode.
- Train only the shadow module during warmup while the main model remains
  unchanged.
- Use per-example-scaled answer injection gradients plus current result logits
  as the shadow input state.
- Compare induced model gradients with the boundary-target ceiling on both fit
  and heldout splits.

## Result

```text
online_mlp_shadow_feedback_stage0b_partial_alignment_no_clean_gate
```

Hidden size `64` reached heldout result/upstream cosines `0.7167/0.7601`, but
train-heldout gaps were `0.2683/0.2202`, above the planned gate. Hidden size
`16` reduced the result gap but heldout result cosine fell to `0.6255`.

## Decision

Do not launch Stage 1 from these simple online MLP warmups. Next work should
change the shadow-generalization mechanism, not rerun the same warmup.
