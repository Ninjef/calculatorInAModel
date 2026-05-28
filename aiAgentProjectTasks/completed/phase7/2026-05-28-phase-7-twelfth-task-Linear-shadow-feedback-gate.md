# Phase 7 Twelfth Task: Linear Shadow-Feedback Gate

## Mission

Test the smallest learned shadow-gradient branch after plain boundary feedback
failed Stage 1 discovery.

## Claim Tested

Can a linear shadow map, fit once from answer-loss calculator-injection
gradients to boundary result-logit gradients, produce a frozen feedback channel
that discovers natural `0..19` result requests without recomputing boundary
targets during training?

## Decision

```text
linear_shadow_feedback_stage0_alignment_pass_stage1_early_lift_negative
```

Stage 0 passed strongly at the model-update level, but the 200-step Stage 1
early-lift smoke failed to beat the previous boundary-feedback baseline.

## Stop Rule

Do not run an 800-step continuation from this exact setup. The next allowed
branch needs a heldout-validated or online-trained shadow module and must clear
an early Stage 1 lift gate before long-run budget.
