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
forced-random).

A follow-up low-LR diagnostic showed the miss was partly source-maturity
limited: continuing the op29 source for `90` steps at `lr=0.0003` with margin
weight `0.1` raised source calc to `0.8211` and final source eval to `0.8233`.
The trusted handoff improved to `0.9067` final / `0.8978` step-600 normal, with
very low controls (`0.0122` injection-zero, `0.0111` forced-random at step
`600`). The calculator path remains causal, but the recovery is still
prescriptive and expensive.

A source-capacity diagnostic changed the picture more sharply. Adding a hidden
result head (`--calculator-result-head-hidden-size 64`) to the same op29
product forced-margin source reached `0.9978` final source eval and produced a
perfect trusted handoff (`1.0000` final / step-600 normal) with low controls.
The result head grew from `7,611` to `12,091` trainable parameters. A fresh
seed repeated the result: source step `630` reached `0.9967`, and the trusted
handoff again reached `1.0000` final / step-600 normal with low controls
(`0.0344` injection-zero, `0.0111` forced-random at step `600`).

## What Should Stop

Do not treat the op19 forced-margin recipe as range-scalable merely because it
cleared product parity. Do not rerun op29 with the same source/handoff seed,
repeat the same low-LR continuation ladder, or jump to op49 with the same
full-grid hard-assignment recipe as novelty. Also do not rerun either completed
op29 `rhead64` seed as novelty.

Also do not respond by tuning local forced-margin knobs. The failure is aligned
with the strategic bottleneck: source acquisition under larger candidate/result
spaces, not downstream readout or product-decoder wiring.

## What Deserves Compute

Useful next work should make range scaling more true:

- Change source acquisition so it is not just full-grid hard assignment plus
  true-result forced-margin pressure.
- Reduce assignment cost with a declared scalability hypothesis and compare to
  the exact-grid ceiling.
- If staying staged, validate whether the hidden result-head capacity fix
  survives larger ranges, many-calculator cost, or lower-cost assignment.

## Are We Closer?

Yes. The staged benchmark is now strong for op19 and wider product decoders,
and op29 can clear with a larger result policy head across two seeds. The
remaining gap is scalability/prescriptiveness: the successful op29 recipe still
uses full-grid hard assignment, true-result forced-margin pressure, and extra
per-calculator source-head capacity. Future agents should validate larger
generality or reduce the assignment/capacity cost rather than repeating op19 or
the same op29 runs.
