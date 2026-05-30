# 2026-05-30 Forced-Margin Benchmark Direction Review

## Why This Review Exists

Automated one-negative forced-margin recovery is now the strongest staged
transfer benchmark in the repo. It has enough positive evidence that future
agents may keep expanding it locally instead of returning to the final
research problem: scalable, non-prescriptive credit assignment.

This review consolidates the post-recovery stability and scale evidence so
the benchmark remains useful without becoming another rerun loop.

## What Changed

The benchmark got stronger:

- Manual low-LR recovery raised one-negative forced-margin handoff to `0.8700`
  final / `0.9050` step-600 normal.
- The first automated fresh-seed recovery raised source calc `0.5825 -> 0.8825`
  during the late window and reached `0.9875` final / `0.9800` step-600 normal
  under trusted frozen-policy additive handoff.
- A second fresh-seed stability check replicated the late recovery mechanism
  and cleared the gate, but with variance: `0.8975` final / `0.9050` step-600
  normal.
- A wider-model stress using an existing `n_embd=32`, `n_head=2` semantic
  decoder was strongly positive: source final eval `0.9125`, trusted handoff
  `1.0000` final / `1.0000` step-600 normal, zero-injection `0.0625`,
  forced-random `0.0325`, and learned calc `0.8850`.

## Belief Changes

One-negative forced-margin plus late recovery is a real staged-transfer
benchmark. It survives a second fresh seed, has visible seed variance, and can
scale to a wider architecture in the tested setup.

But the benchmark is still not the thesis result. It uses:

- hard improvement assignment during source acquisition;
- true-result forced-margin pressure;
- a pre-trained semantic decoder;
- frozen-policy transfer into the non-bottleneck model.

The wider-model result is encouraging for scale, but has a non-product decoder
caveat because the available wider semantic decoder used
`answer_decoder_interaction=none`, not the later product decoder.

## What Should Stop

Do not treat these as novel:

- more seed-only replications at the same tiny architecture unless predeclared
  as part of a small-N stability estimate;
- local forced-margin start-step, negative-count, margin, LR, or recovery
  length sweeps;
- repeating the same `n_embd=32`, `n_head=2` non-product scale stress;
- cheap selector/proxy work around the forced-margin benchmark;
- claims that high frozen-policy handoff solves non-prescriptive discovery.

## What Deserves Compute

Forced-margin compute is now justified only when it answers a broader question:

- product-decoder parity for the wider model;
- larger operand ranges or a genuinely larger architecture family;
- many-calculator-style cost or interference stress;
- replacing hard assignment or true-result forcing while comparing against this
  benchmark;
- a predeclared stability estimate with enough seeds to change confidence, not
  a single nearby rerun.

Otherwise, mainline compute should move toward less-prescriptive
answer-derived target construction, scalable assignment approximation, or
credit-assignment mechanisms that can train from answer-derived signals.

## Are We Closer To The Goal?

Yes, for staged non-bottleneck viability and scale.

The project has a stronger demonstration that a learned calculator policy can
be transferred into a non-bottleneck path and used causally, and that the
recipe is not limited to the tiniest architecture tested.

No, for the central thesis.

The result still depends on prescriptive source targets, candidate scoring,
and frozen transfer. It does not prove from-scratch answer-loss-only
calculator discovery, scalable many-calculator training, or model-decided
calculator use.

## Steering Decision

Treat automated forced-margin recovery as the benchmark to beat, not the next
knob branch.

Future work should either stress a new axis that matters for the thesis
(`product decoder`, `larger operand range`, `larger architecture`,
`many-calculator cost`) or remove prescriptiveness by replacing hard assignment
or true-result forcing. If an idea does neither, write a review or pivot rather
than spending compute.
