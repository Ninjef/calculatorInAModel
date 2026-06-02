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

## Cadence Follow-Up

Lower-frequency full-memory prior fitting partially reduces this bottleneck,
but simple cadence thinning has a sharp quality limit.

Every-10 fitting cut prior updates to `501`, but underfit the prior and missed
the benchmark: source overall `0.9475`, train `0.978125`, heldout `0.7625`,
prior train/heldout `0.953125`/`0.7875`.

Every-2 fitting preserved the source gate with half the prior updates: overall
`0.9950`, train `1.0000`, heldout `0.9125`, heldout controls low, prior
train/heldout `1.0000`/`0.9125`, and `2501` prior updates instead of `5001`.
Its trusted frozen-policy additive handoff remained causal and high:
`395/400 = 0.9875`, diagnostic calculator-result accuracy `0.984375`, and
128-sample controls `0.015625` injection-zero, `0.0078125` forced-zero,
`0.0078125` forced-random.

Steering update: do not run a cadence ladder as novelty. The safe benchmark is
now every-2; every-10 shows update starvation. The next scalable mechanism
should fit until the prior/memory has converged, refresh after memory changes,
or use coreset/reservoir batches to beat `2501` full-memory updates without
losing the heldout/handoff gate.

## Convergence-Stop Follow-Up

Sustained train-memory convergence is a better cost reducer than cadence alone,
but the stopping signal has to be conservative.

Stopping at the first prior train-memory accuracy `1.0` saved many updates
(`1029`), but it stopped too early for heldout generalization: source overall
fell to `0.9825`, heldout exact/calc to `0.8750`, and prior heldout accuracy to
`0.8750`. This disproves "first train fit means prior is ready."

Requiring `100` converged fit updates preserved the benchmark while reducing
cost: source overall `0.9950`, train `1.0000`, heldout `0.9125`, controls low,
prior train/heldout `1.0000`/`0.9125`, and only `1889` prior updates versus
`2501` for every-2 and `5001` for full-fit. The trusted handoff also passed:
`397/400 = 0.9925`, diagnostic calc `0.984375`, with 128-sample controls
`0.0546875` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random.

Steering update: patience-100 is the new safe train-convergence benchmark, but
do not run patience ladders as novelty. The next mechanism should use a
validation/heldout-prior signal or coreset/reservoir fitting to beat `1889`
updates while preserving the same source/handoff gate.

## Random-Coreset Follow-Up

Uniform random half-memory prior fits do not preserve the gate.

Using `--result-boundary-target-amortized-prior-fit-batch-size 160` with the
every-2 patience-100 recipe reduced examples per prior fit, but the prior never
converged: final train/heldout prior accuracy was only `0.909375`/`0.7750`,
the stop rule never activated, and prior updates remained `2501`. The source
kept train exact/calc high (`0.996875`) but missed heldout (`0.8125`) and
overall (`0.9675`), with controls still low.

Steering update: do not run random fit-batch-size ladders as novelty. The
negative is consistent with the earlier batch64 underfit. Any coreset/reservoir
next step needs to be structured, coverage-aware, or validated by a heldout
prior signal, not another uniform random batch size.

## Structured-Coreset Follow-Up

Target-stratified half-memory prior fitting is a positive structured coreset
result.

Changing only the half-memory prior fit sampler from random to
target-stratified, while keeping fit batch size `160`, every-2 fitting, and
patience-100 stopping, restored and slightly exceeded the heldout source gate:
overall `0.9900`, train `0.996875`, heldout `0.9375`, heldout controls
injection-zero `0.0125`, forced-zero `0.0000`, forced-random `0.0000`.
Prior train/heldout accuracy was `0.965625`/`0.9000`. Prior updates remained
`2501`, so the stop rule did not activate, but forced-result evals fell to
`67,584` versus `86,016` for the random-half and full-memory comparators.

The trusted frozen-policy additive handoff also passed from this source:
final eval `0.9975`, diagnostic exact/calc `0.9921875`/`1.0000`, routed
calculator-result accuracy `1.0000` for all four hooks, and final 128-sample
counterfactuals low from the handoff metrics: injection-zero `0.0000`,
forced-zero `0.0546875`, forced-random `0.0390625`.

Steering update: target-stratified half-memory is now the structured coreset
benchmark to beat. Do not rerun random batch-size ladders. The next useful
cost-reduction step should combine target-stratified sampling with a
validation/prior convergence stop, or stress it on a fresh seed/range axis
before promoting it to the default recipe.

## Validation-Stop Follow-Up

Validation-heldout stopping does not currently improve the target-stratified
cost/quality tradeoff.

The tested recipe held out `20%` of prompt-memory entries from prior fitting,
fit target-stratified batch `160`, stopped on validation accuracy `>=0.9` for
`100` fit steps, and stopped at `2359` prior updates. It reduced forced-result
evals to `53,760`, but the source missed the heldout gate: overall `0.9725`,
train `0.990625`, heldout `0.8625`, train/heldout prior
`0.98125`/`0.8625`. This is below both the target-stratified source
(`0.9375` heldout) and the sustained full-memory benchmark (`0.9125`
heldout). No handoff was run.

Steering update: do not run validation-heldout threshold or patience ladders as
novelty. The likely failure is excluding memory entries from the prior fit, not
merely picking the wrong threshold. If validation is used again, make it
eval-only while fitting all entries, or use a rolling/full-fit stopping signal.
Otherwise spend compute on stressing the positive target-stratified coreset
across seed/range.

## Eval-Only Validation Follow-Up

Eval-only validation stopping is now the target-stratified branch's strongest
prior-fit cost-reduction result, with seed-sensitivity and total-cost caveats.

Adding a validation mode that keeps all prompt-memory entries in the fit pool
and uses the deterministic validation split only for metrics/stopping reversed
the validation-heldout miss. On effective seed13, with target-stratified batch
`160`, validation fraction `0.2`, stop metric `validation_accuracy`, threshold
`0.9`, and patience `100`, the source stopped at step `3250` with `1613` prior
updates. Source metrics were overall `0.9825`, train `0.99375`, heldout
`0.9500`, prior train/heldout `0.978125`/`0.9500`, and low heldout controls.
The trusted frozen-policy additive handoff reached final `1.0000`, diagnostic
exact/calc `1.0000`/`0.953125`, and low final counterfactuals (`0.0078125`
injection-zero, `0.0390625` forced-zero, `0.015625` forced-random).

A same-effective-seed11 isolation run also cleared the source/handoff gate but
was less strong: source overall `0.9725`, train `0.984375`, heldout `0.9125`,
prior train/heldout `0.9625`/`0.9125`, and `1784` prior updates. Its trusted
handoff reached final `1.0000`, diagnostic exact/calc `1.0000`/`0.9453125`,
and low counterfactuals (`0.0000` injection-zero, `0.0703125` forced-zero,
`0.046875` forced-random).

Steering update: eval-only validation, not validation-heldout fitting, is the
next prior-update cost-reduction lead. It cuts prior updates below the
sustained full-memory benchmark on both tested effective seeds (`1613`/`1784`
vs `1889`) while preserving trusted handoff. But source quality is
seed-sensitive, and forced-result evals rose because prompt memory filled at
step `100` instead of the target-stratified seed11 benchmark's step `50`
(`89,088` same-seed, `124,416` seed13, versus `67,584`). Next work should
stress this result on a larger range and diagnose memory-fill/forced-eval cost
before treating it as the default scalable recipe.

## Op29 Range Follow-Up

The larger-range stress is a mixed negative for the constant-batch recipe, not
a reason to rerun local stopping knobs.

At `operand_max=29`, the four-hook shared-output eval-only target-stratified
source used the same constant prior fit batch `160` over `720` train prompts
and `180` heldout prompts. It filled prompt memory by step `200` after
`290,304` forced-result evals, then continued to `5000` steps. Source train
accuracy was high (`0.9931`), but heldout exact/calc was only `0.8444`,
overall was `0.9622`, prior train/heldout were `0.8375`/`0.7667`, and the
validation stop never fired (`2501` prior updates). No trusted handoff was run
because the source missed the heldout gate.

Post-hoc prior diagnostics split the failure: the discovered train-memory
targets were mostly correct (`0.9931` matched true sums), and memory-fill cost
was not the main blocker. But the h64 numeric prior needed much more full-memory
optimization to fit op29: after `600` full-memory steps it reached only
`0.8958` train / `0.6722` heldout, after `2500` steps it reached `0.9875` /
`0.9000`, and h128 at `2500` reached `0.9889` / `0.9278`.

Steering update: do not promote eval-only target-stratified batch160 as
range-scalable. The next high-leverage experiment should change the prior
capacity/features or fit dynamics, with explicit many-calculator cost
accounting. A proportional-batch source run is only useful if it is framed as a
costed diagnostic against richer/longer prior fitting, not as a batch-size
ladder.

## Op29 H128 Prior-Capacity Follow-Up

Increasing capacity alone does not fix online op29 fitting.

The h128 source kept the same op29 recipe and constant fit batch `160`, only
changing `--result-boundary-target-amortized-prior-hidden-size` from `64` to
`128`. It improved overall exact/calc to `879/900 = 0.9767` and train
exact/calc to `719/720 = 0.9986`, but heldout exact/calc reached only
`155/180 = 0.8611`. The online prior remained the clear bottleneck:
train/heldout prior accuracy was only `0.8097`/`0.7111`, the validation stop
never fired, and prior updates stayed at `2501`; forced-result evals were
`294,912`.

A post-hoc h128 full-memory fit on the same trace reached `0.9944` train /
`0.9278` heldout with target match `0.9986`, so the target table and model
capacity are not the main blockers. The problem is that constant-batch online
target-stratified fitting does not train the prior well enough at op29.

Steering update: do not run hidden-size bumps as novelty. The next experiment
must alter the online fit dynamics themselves, such as a post-memory-fill
full-memory refresh followed by cheaper replay, or a coverage-aware/proportional
fit with explicit cost accounting.

## Op29 Post-Fill Full-Refresh Follow-Up

Changing the online fit dynamics repairs the op29 heldout and handoff gate, but
the repair is expensive.

The new
`--result-boundary-target-amortized-prior-full-refresh-after-memory-full-updates`
mechanism forces full-memory prior updates after prompt memory first fills,
then returns to the configured target-stratified fit batch. With h128,
target-stratified batch `160`, eval-only validation, and `2500` full-refresh
updates after memory full, the op29 source reached:

- overall exact/calc `884/900 = 0.9822`;
- train exact/calc `0.9972`;
- heldout exact/calc `155/180 = 0.9167`;
- prior train/heldout `0.9958`/`0.9167`;
- heldout controls `0.0278` injection-zero, `0.0000` forced-zero,
  `0.0111` forced-random;
- `2755` total prior updates and `294,912` forced-result evals.

The trusted 600-step frozen-policy additive handoff then reached final
`900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9922`, and low controls:
`0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random.

This changes the interpretation of the op29 stress. Constant-batch h64/h128
online fitting was the wrong dynamic, not a sign that the target table or
handoff geometry was fundamentally failing at op29. Post-fill full refresh is
therefore the new positive range-scaling benchmark for this branch.

Steering update: do not rerun op29 batch160, hidden-size bumps, random
fit-batch ladders, validation threshold/patience ladders, or the same
full-refresh pass as novelty. The next useful experiment must reduce or
structure the refresh cost while preserving the same source/handoff gate:
staged full refresh then coreset replay, coverage-aware/proportional fitting,
or a better refresh-stop/freeze transition with explicit many-calculator cost
accounting.

## Op29 Full-Refresh Early-Stop Follow-Up

Simple early stopping during the full-refresh window is not sufficient.

Allowing the existing validation stop rule to end post-memory-fill full refresh
cut prior updates from the positive run's `2755` to `1140`, but source heldout
exact/calc fell from `0.9167` to `0.8167`. The source still trained the seen
prompts well (`0.9889` train exact/calc) and controls stayed low, so this was a
prior generalization miss rather than a wiring failure. No trusted handoff was
run because the source missed the heldout gate.

Steering update: avoid validation threshold/patience ladders or "stop earlier"
variants. The next cost-reduction attempt needs a stronger transition that
keeps coverage: for example staged refresh followed by coreset replay, dual
train+validation/high-confidence stopping, or coverage-aware/proportional
refresh with explicit source/handoff and cost gates.

## Op29 Dual-Guard Refresh Stop Follow-Up

Requiring train-memory prior coverage in addition to validation avoids the
simple early-stop failure, but the update saving is modest.

The new stop guard requires train prior accuracy before validation-based
stopping can end the full-refresh window. With validation `>=0.9`, train
requirement `>=0.98`, and patience `100`, the op29 h128 source reached
`0.9956` overall, `1.0000` train, `0.9667` heldout, and prior train/heldout
`0.9972`/`0.9667`, with low heldout controls. It stopped at `2570` prior
updates, compared with `2755` in the earlier full-refresh positive. The trusted
600-step additive handoff reached `900/900 = 1.0000` with low controls.

Steering update: this validates the diagnosis that validation-only stop was
too optimistic, but it is not yet a major scalability win. Do not run
train-requirement threshold/patience ladders. The next branch should target a
larger cost change: staged refresh plus coreset replay, coverage-aware refresh,
or explicit many-calculator cost accounting.
