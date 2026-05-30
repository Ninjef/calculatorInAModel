# 2026-05-30 Forced-Margin Range-Stress Review

## What Changed

The staged forced-margin benchmark survived wider product-decoder parity at
`operand_max=19`, but did not cleanly survive the first larger-range stress.
For `operand_max=29`, the oracle product decoder was not the issue: it reached
full-grid `1.0000` eval. The source policy was the bottleneck.

The op29 automated source improved during late recovery from `0.3533` at step
`600` to `0.6889` at step `630`, with final source eval `0.7133`. The trusted
frozen-policy additive handoff reached `0.8533` final / `0.8278` step-600
normal, with low step-600 controls (`0.0344` injection-zero, `0.0189`
forced-random). So the calculator path remains causal, but the high gate is
not cleared.

## What Should Stop

Do not treat the op19 forced-margin recipe as range-scalable merely because it
cleared product parity. Do not rerun op29 with the same source/handoff seed,
or jump to op49 with the same full-grid hard-assignment recipe as novelty.

Also do not respond by tuning local forced-margin knobs. The failure is aligned
with the strategic bottleneck: source acquisition under larger candidate/result
spaces, not downstream readout or product-decoder wiring.

## What Deserves Compute

Useful next work should make range scaling more true:

- Change source acquisition so it is not just full-grid hard assignment plus
  true-result forced-margin pressure.
- Reduce assignment cost with a declared scalability hypothesis and compare to
  the exact-grid ceiling.
- If staying staged, test whether materially more source capacity or a
  different recovery objective changes the op29 failure mode before going to
  op49.

## Are We Closer?

Yes, but mostly by clarifying the boundary. The staged benchmark is now strong
for op19 and wider product decoders, but op29 shows it is not yet a scalable
range solution. Future agents should treat range scaling as unresolved and
should avoid repeating op19 success conditions.
