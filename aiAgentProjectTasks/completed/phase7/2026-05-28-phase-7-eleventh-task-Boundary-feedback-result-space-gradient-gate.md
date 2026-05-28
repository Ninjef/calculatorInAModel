# Phase 7 Eleventh Task: Boundary-Feedback Result-Space Gradient Gate

## Mission

Test an explicitly biased backward channel for natural `0..19` result-space
calculator requests.

This is not an oracle/readout check, target-off retention rerun, vanilla
policy-gradient run, raw exact expected-cost run, or decoder-only calibration
branch.

## Claim Tested

Can answer-loss gradients at the calculator boundary, projected back into
result-space logits, produce result-proj/upstream updates that align with the
exact-grid boundary-target ceiling and then train a hard result request with
semantic decoder frozen and all true-result, oracle-action, boundary-target,
aux, expected-cost, and anchor objectives off?

## Required Gate

Use exhaustive `20 x 20` grid, `calculator_action_head=result_space`, frozen
semantic decoder, and upstream open.

Pass requires:

- nonzero result-proj and upstream gradients;
- semantic decoder gradient exactly `0.0`;
- positive result-proj and upstream cosine versus the boundary-target ceiling.

## Decision

```text
boundary_feedback_stage0_output_projection_alignment_pass_stage1_discovery_negative
fixed_random_direct_feedback_stage0_result_head_alignment_negative
```

The output-projection feedback channel passed Stage 0 but failed Stage 1
discovery. The fixed-random direct-feedback seed tested did not pass the
Stage 0 result-head cosine gate, so no fixed-random long run was launched.

## Follow-up

Do not spend more mainline budget on the same plain output-projection feedback
schedule. The next allowed branch is a learned shadow-gradient or stronger
feedback module that first passes the same Stage 0 gate and then shows early
Stage 1 lift above this `0.16` final exact-match baseline.
