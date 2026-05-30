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

The next larger-range check is more sobering. At `operand_max=39`, the product
oracle decoder again cleared (`1600/1600 = 1.0000`), but the exact full-grid
`rhead64` source run was interrupted after about `33` local CPU minutes with
checkpoints only through step `540`; step `540` eval was `0.543`. A bounded
90-step continuation recovered source eval to `0.940`, and the trusted handoff
reached `0.9475` final / `0.9419` step-600 normal with low controls
(`0.0000` injection-zero, `0.0138` forced-random). This is causal transfer, but
not the perfect op29 gate and not a scalable training story.

## What Should Stop

Do not treat the op19 forced-margin recipe as range-scalable merely because it
cleared product parity. Do not rerun op29 with the same source/handoff seed,
repeat the same low-LR continuation ladder, jump to op49 with the same
full-grid hard-assignment recipe as novelty, or rerun either completed op29
`rhead64` seed. Also do not rerun the same op39 full-grid step-540 continuation
and handoff path as novelty.

Also do not respond by tuning local forced-margin knobs. The failure is aligned
with the strategic bottleneck: source acquisition under larger candidate/result
spaces, not downstream readout or product-decoder wiring.

## What Deserves Compute

Useful next work should make range scaling more true:

- Change source acquisition so it is not just full-grid hard assignment plus
  true-result forced-margin pressure.
- Reduce assignment cost with a declared scalability hypothesis and compare to
  the exact-grid ceiling.
- If staying staged, validate many-calculator cost or lower-cost assignment
  before more larger-range full-grid runs.

## Are We Closer?

Yes, but the warning is sharper. The staged benchmark is strong for op19 and
wider product decoders, op29 clears with a larger result policy head across two
seeds, and op39 can transfer causally after source continuation. The remaining
gap is scalability/prescriptiveness: op39 already shows high local CPU cost and
sub-perfect handoff while still using full-grid hard assignment, true-result
forced-margin pressure, and extra per-calculator source-head capacity. Future
agents should reduce assignment/capacity cost rather than repeating op19/op29
or pushing full-grid range alone.
