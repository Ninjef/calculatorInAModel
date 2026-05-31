# 2026-05-31 Online Hard Memory Result-Boundary Review

## Trigger

The previous cached-teacher steering review concluded that hard targets train
well, but cached/full-enum teacher tables are not a scalable recipe. The online
hard-memory result-boundary branch is a direct mechanism-level response to
that lesson.

## What Changed

Online sparse zero-improvement scoring can discover a clean hard target memory
on the op19 fixed grid. With topk8+unique24 candidates, the memory reached
full coverage and `best_true=1.0000` by step 50. Training against that hard
memory reached `0.9675` learned calculator accuracy and `0.9725` final eval by
step 800.

The freeze-when-full variant shows the expensive scoring does not have to
continue after discovery on this gate: cumulative forced-result evaluations
stopped at `86,400`, yet the final result matched the continuously rescored
branch.

## What Should Stop

- Do not run more same-seed op19 online-hard-memory length/LR repeats.
- Do not compare more soft target temperatures against this branch unless the
  target construction changes materially.
- Do not claim the fixed-grid memory result proves streaming scalability.

## What Deserves Compute

1. Fresh-seed replication of the source gate.
2. Trusted additive handoff from the step-800 source.
3. Streaming/fresh-prompt validation where prompt-keyed memory cannot simply
   memorize a closed train grid.
4. Many-calculator/routed-hook accounting once source and handoff replication
   are real.

## Strategic Status

This is now the main less-prescriptive answer-derived lead. It is not solved,
because fixed-grid hard memory can be transductive. But unlike prior sampled
zero-improvement, cached teacher, and critic/proposal branches, it combines a
plausibly cheaper sparse discovery phase with hard-target source acquisition
that reaches mature calculator accuracy.
