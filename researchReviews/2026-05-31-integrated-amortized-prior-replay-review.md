# 2026-05-31 Integrated Amortized-Prior Replay Review

## Why This Review

The prompt-keyed heldout split exposed a serious failure: online hard memory
solved seen prompts but did not generalize to prompts absent from memory. The
numeric prior diagnostic was promising, but only post-hoc. This review decides
whether the integrated source-training result changes the strategic picture or
just adds another local replay tweak.

## What Changed

Integrated numeric-prior replay is now a real source-training positive.

With the old online prior fit path, the 5000-step op19 heldout source improved
heldout prompts from the prompt-memory-only baseline `0.0875` to `0.7125`, but
the online prior itself was weak (`0.7000` heldout accuracy). An offline
full-batch prior fit from the same final train trace reached `0.9125` heldout,
showing target quality was present but the online prior fit was the bottleneck.

Decoupling the prior fit batch from model replay and using full-memory prior
fit during source training closed that gap:

- source overall exact/calc `398/400 = 0.9950`;
- train exact/calc `320/320 = 1.0000`;
- heldout exact/calc `73/80 = 0.9125`;
- heldout controls low: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`;
- online prior heldout accuracy `0.9250`;
- prompt memory covered only the `320` train prompts.

The trusted frozen-policy additive handoff from that source also passed:

- final eval `400/400 = 1.0000`;
- controls low: injection-zero `0.0234`, forced-zero `0.0078`,
  forced-random `0.0156`;
- diagnostic calculator-result accuracy `0.984375`.

## What Should Stop

- Do not rerun prompt-keyed memory heldout without a non-transductive target
  mechanism.
- Do not treat minibatch prior fit with the replay batch size as sufficient;
  the A/B says it underfits the prior and leaves heldout at `0.7125`.
- Do not claim full-memory prior fit is already scalable. It works, but its
  cost grows with prompt-memory size.
- Do not spend mainline compute on another same-seed op19 full-memory repeat as
  novelty.

## What Deserves Compute

- Cheaper prior fitting that preserves the full-memory fit quality: cached
  prior refreshes after memory fill, multiple prior updates only when memory
  changes, reservoir/coreset memory batches, or lower-frequency full-memory
  sweeps.
- Fresh-seed replication only after the cheaper prior-fit story is clear, or
  if the source/handoff result unexpectedly fails under a cost-reduced variant.
- Many-calculator accounting for the new method: forced-result scoring is now
  bounded by train-prompt memory fill, but prior fitting and replay add their
  own per-calculator costs.

## Are We Closer?

Yes. This is the first source run in the current branch where prompts absent
from hard memory learn calculator-result behavior during source training, and
the resulting policy transfers into the non-bottleneck additive setting.

But the goal is not finished. The method still uses answer-derived sparse
forced-result scoring on train prompts and full-memory prior fitting to make
fresh-prompt replay stable. The next strategic step is not another local
accuracy repeat; it is turning the full-memory prior-fit stabilizer into a
scalable approximation.

## Steering Decision

Treat integrated numeric-prior replay with full-memory prior fit as the new
positive benchmark for fresh-prompt source acquisition and trusted handoff.
Mainline work should now reduce the prior-fit cost while preserving heldout
source accuracy and handoff causality.
