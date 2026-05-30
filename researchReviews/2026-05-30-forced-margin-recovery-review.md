# 2026-05-30 Forced-Margin Recovery Review

## Why This Review Exists

The prior forced-margin branch review allowed only two strategically meaningful
continuations: a predeclared source-recovery test and a fresh-seed stability
or automation test. Both have now been run, and the result is strong enough
that future agents may be tempted to keep tuning the branch.

This review decides what changed, what should stop, and how to use the result
without confusing a strong staged-transfer recipe for the final project goal.

## What Changed

The source-policy maturity hypothesis was correct.

- Manual recovery from the longer one-negative forced-margin step-600 source
  checkpoint raised source calc from `0.5225` to `0.7725` in 30 low-LR steps
  and improved trusted frozen-policy handoff from the unrecovered
  `0.7330-0.7400` range to `0.8700` final / `0.9050` step-600 normal.
- Automated recovery added an in-run late forced-margin weight override and
  replicated on a fresh seed. The source improved from `0.5825` at step `600`
  to `0.8825` at step `630`, and trusted frozen-policy handoff reached
  `0.9875` final / `0.9800` step-600 normal with low injection-zero and
  forced-random controls.
- The automated forced-margin handoff is now the strongest forced-margin
  staged-transfer result and exceeds the previous automated scheduled
  forced-true recovery handoff (`0.9400` final).

## Belief Changes

One-negative forced-margin plus late low-LR recovery is a strong source
acquisition recipe for staged transfer. It is no longer merely a bounded
auxiliary that fails to beat scheduled forced-true; with recovery, it can
produce excellent non-bottleneck handoff.

But this does not solve the central learning problem. The branch still uses:

- hard improvement assignment for the bottleneck source policy;
- true-result forced-margin pressure;
- frozen-policy staged handoff into non-bottleneck mode.

So the result is important evidence about transfer geometry and source-policy
maturity, not proof of scalable, non-prescriptive calculator discovery.

## What Should Stop

Do not treat the following as novel:

- rerunning the seed-15 manual recovery checkpoint;
- rerunning the seed-16 automated recovery plus handoff;
- tuning forced-margin start step, margin value, negative count, or late
  recovery length on the same setup;
- adding more source-accuracy or forced-loss trigger thresholds without a new
  signal family;
- claiming the branch solves the project goal because the non-bottleneck
  handoff is high.

The branch has answered its local question: late recovery rescues and
strengthens one-negative forced-margin source acquisition.

## What Deserves Compute

Forced-margin compute is justified only if it answers a broader question:

- Stability/scale: verify the automated recipe across several fresh seeds,
  larger operand ranges, or a meaningfully larger model, with the same trusted
  handoff controls.
- Lower prescriptiveness: replace true-result forced-margin pressure with a
  target construction or estimator that discovers useful result pressure from
  answer loss.
- Scalability of assignment: reduce or approximate hard improvement assignment
  while comparing against the exact-grid assignment ceiling.

Otherwise, mainline effort should pivot back to scalable credit assignment:
local-target estimator changes, different target construction, or learned
proposal mechanisms with explicit streaming/generalization validation.

## Are We Closer?

Yes, in a staged-transfer sense.

The project now has a stronger working non-bottleneck recipe and clearer
evidence that late gentle recovery can convert a promising source policy into
usable transfer geometry. This is valuable for proving the architecture can be
made to work.

No, in the final-goal sense.

The result remains prescriptive and full-grid/source-scaffolded. It does not
prove answer-loss-only discovery, scalable many-calculator training, or
non-bottleneck calculator discovery without freezing a prescriptive source
policy.

## Steering Decision

Treat automated one-negative forced-margin recovery as the current best
staged-transfer source recipe and as a benchmark for future source objectives.

Do not continue local forced-margin knob tuning. Future work should either
stress the recipe at scale/stability or use it as a bridge toward less
prescriptive credit assignment.

The next strategically valuable question is not "can this seed get even
higher?" It is:

```text
Can we preserve this transfer behavior while removing hard assignment or
true-result forcing?
```
