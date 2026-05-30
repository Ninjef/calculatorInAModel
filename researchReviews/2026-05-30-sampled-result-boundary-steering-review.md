# 2026-05-30 Sampled Result-Boundary Steering Review

## Trigger

The project tested whether the policy-topk+unique24 candidate recipe, which
works for hard improvement assignment, transfers to the less-prescriptive
answer-derived result-boundary source objective.

## What Changed

- Candidate-scored result-boundary source training is implemented and tested.
- The topk8+unique24 gate reduced forced scoring from `39/39` to `24/39`
  result classes per prompt.
- Candidate coverage was not the main failure: true-candidate coverage reached
  `0.9600` by step `200`.
- Source learning was still weak: step-200 learned-best/source calc was
  `0.3425`, snapshot normal/calc `0.3675`, and final eval `0.3525`.
- Matched full-enum hard-best result-boundary comparators were stronger
  (`0.5450`/`0.5475` and `0.4625`/`0.4225` learned-calc/final-eval pairs in
  nearby gates).

## Stop

- Do not ladder sampled result-boundary target counts, top-k counts, unique
  sampling, or seed-only reruns around this exact mechanism.
- Do not treat candidate coverage alone as the result-boundary source bottleneck.
- Do not conflate policy-topk success for hard assignment with success for
  answer-derived result-boundary targets.

## Continue

Result-boundary remains a useful answer-derived bridge, but the next attempt
needs a mechanism change:

- active proposal/training co-design where candidate proposals adapt to the
  policy's learning dynamics;
- stronger online/state-calibrated proposals with a gate that beats the current
  topk8+unique24 source result at lower cost;
- or a different less-prescriptive credit-assignment family.

## Decision

```text
sampled_result_boundary_policy_topk_branch_paused
```

This branch gave a useful answer to the user's zoom-out concern: the sampled
result-boundary candidate gate was a real method test, but its negative result
means future agents should not spend more turns on small local variants.
