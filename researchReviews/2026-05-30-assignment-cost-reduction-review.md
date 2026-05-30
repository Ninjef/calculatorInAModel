# 2026-05-30 Assignment-Cost Reduction Review

## Why This Review Exists

The op39 `rhead64` range stress made full-grid source cost concrete, so the
next question was whether hard improvement assignment can be made cheaper
without abandoning the staged benchmark. A first direct test of uniform sampled
hard assignment is now complete and should update future direction.

## What Changed

We added sampled hard-assignment support and compared exact assignment against
sample16 and sample32 on the op19 `rhead64` forced-margin source gate.

| Assignment | Scored results | Step-200 true coverage | Step-200 target acc | Best snapshot normal | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.9900` | `0.8625` | `0.7350` |
| Sample16 | `16/39` | `0.6125` | `0.4581` | `0.3650` | `0.3525` |
| Sample32 | `32/39` | `0.7400` | `0.6773` | `0.4050` | `0.3800` |

Runtime dropped only modestly at this scale: about `115s` exact, `88s`
sample16, and `106s` sample32. The sampled branches did not preserve the
source signal well enough to justify that savings.

We then added exact assignment refresh cadence as a second cost-reduction
mechanism. This preserved full coverage on refresh steps but reused cached
targets between them.

| Assignment cadence | Full refreshes over 201 steps | Best snapshot normal | Final eval | Approx wall time |
| --- | ---: | ---: | ---: | ---: |
| Exact every step | `201` | `0.8625` | `0.7350` | `115.5s` |
| Refresh every 2 | `101` | `0.5875` | `0.5925` | `106.4s` |
| Refresh every 5 | `41` | `0.4950` | `0.4950` | `105.1s` |

Refresh cadence is less destructive than uniform sparse candidates, but it
still fails to preserve exact source acquisition and does not show meaningful
full-run wall-clock savings in this diagnostic setup.

Finally, we tested duplicate-free sampled candidates. This is a coverage-aware
variant of the sampled assignment gate: the learned result is included, and the
remaining candidates are sampled without replacement per prompt.

| Assignment | Scored results | Step-200 true coverage | Best snapshot normal | Final eval |
| --- | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.8625` | `0.7350` |
| Sample32 duplicate-prone | `32/39` | `0.7400` | `0.4050` | `0.3800` |
| Unique32 | `32/39` | `0.9275` | `0.6250` | `0.6100` |

Duplicate removal clearly helps, but unique32 still scores most of the result
vocabulary and remains below the exact ceiling.

The first non-uniform proposal is different. Reserving 8 candidate slots for
the model's current result-policy top-k classes and filling the rest with
unique random candidates recovered much more of the exact signal:

| Assignment | Scored results | Step-200 true coverage | Best snapshot normal | Final eval |
| --- | ---: | ---: | ---: | ---: |
| Exact | `39/39` | `1.0000` | `0.8625` | `0.7350` |
| Topk8+unique16 | `16/39` | `1.0000` | `0.6850` | `0.6725` |
| Topk8+unique24 | `24/39` | `1.0000` | `0.7725` | `0.7500` |
| Topk8+unique32 | `32/39` | `1.0000` | `0.7925` | `0.8600` |

This is the first assignment-cost proposal worth validating further. It is not
a solved scalable recipe yet because the result is one op19 source gate, but it
changes the next direction from "uniform coverage is insufficient" to
"policy-aware proposals can preserve target coverage at lower scorer count."

The first staged validation is positive. Running `topk8+unique24` for the
longer `630`-step op19 `rhead64` source recipe with late recovery reached
`1.0000` source final eval; the trusted frozen-policy additive handoff from
step `630` also reached `1.0000` final eval / step-600 normal with low controls
(`0.0200` injection-zero, `0.0325` forced-random). This upgrades the proposal
from local source-screen positive to op19 staged-transfer positive, while still
leaving fresh-seed/range/many-calculator validation open.

Fresh-seed validation is also positive. Repeating the same staged recipe on CLI
seed `45` / effective seed `47` again reached `1.0000` source final eval and
`1.0000` trusted handoff final eval. The step-600 handoff controls stayed
causal: injection-zero `0.0475`, learned calc `1.0000`, and forced-random
`0.0250`. That makes the proposal a replicated op19 transfer result, but not
yet a range-scaled, many-calculator, or non-prescriptive method.

The first range validation is positive. On the op29 `rhead64` exact-ceiling
seed, `topk8+unique24` scored `24/59` result classes and still reached
`1.0000` source final eval plus `1.0000` trusted handoff final eval. Step-600
handoff controls were causal: injection-zero `0.0356`, learned calc `1.0000`,
and forced-random `0.0189`. This is a real range/cost improvement over exact
assignment on that seed, though it remains one op29 seed and still uses hard
assignment, forced-margin source shaping, hidden result-head capacity, a
pretrained decoder, and frozen transfer.

The op29 range result now replicates. On CLI seed `31` / effective seed `33`,
matched to the exact fresh-range comparator, the sparse source reached
`899/900 = 0.9989` and the trusted handoff reached `900/900 = 1.0000`.
Step-600 handoff controls stayed causal: injection-zero `0.0333`, learned calc
`0.9989`, and forced-random `0.0111`. Further op29 seed replications are now
low value; the next axis should change.

## What Should Stop

- Uniform sampled hard-assignment count ladders on the same op19 source gate.
- Treating high nominal sample count as enough. Sample32 scored most of the
  result vocabulary but still covered the true result only `0.7400` of prompts
  at step `200`, partly because uniform sampling duplicates candidates.
- Jumping from this failure to larger full-grid range runs without a changed
  assignment-cost hypothesis.
- Fixed refresh-interval ladders on the same op19 `rhead64` gate. Stale exact
  targets are not enough without an adaptive freshness or predictive update
  mechanism.
- Unique-uniform sample-count ladders on the same gate. Unique32 is the useful
  diagnostic point: coverage helps, but it is still not enough.
- Topk8 unique count ladders on the same gate. The useful threshold is already
  mapped at 16/24/32.
- The same effective-seed-43 op19 topk8+unique24 source630 plus handoff600
  path. It cleared; repeat only with a new validation axis.
- More op19 `rhead64` topk8+unique24 fresh-seed source630 plus handoff600
  replications. Two seeds have cleared; the next axis should be about
  scalability or prescriptiveness.
- The same effective-seed-29 op29 `rhead64` topk8+unique24 source630 plus
  handoff600 path. It has cleared against the exact-grid ceiling.
- More op29 `rhead64` topk8+unique24 seed replications. Effective seeds `29`
  and `33` both cleared; future work needs many-calculator, op39, or
  prescriptiveness pressure.

## What Deserves Compute

- Validate policy-aware top-k proposals beyond replicated op29 transfer:
  op39/many-calculator cost or reduced prescriptiveness.
- More active/non-uniform proposals only if they beat the topk8+unique baseline
  or explain when/why top-k coverage fails.
- Structured proposals that exploit arithmetic/result geometry while still
  being validated against exact-grid assignment ceilings.
- Non-enumerative credit signals that avoid hard assignment rather than
  sampling it thinly.
- Adaptive target-refresh criteria that can skip scoring only when target
  freshness is likely preserved, and that report real compute savings.
- Many-calculator accounting, but only when paired with a candidate mechanism
  whose single-calculator ceiling comparison is not already negative.

## Are We Closer?

Meaningfully closer on the scalable-assignment axis, but not done. This did not
solve non-prescriptive credit assignment, but it closed tempting cheap paths and
turned policy-aware sparse assignment into a replicated op19 and replicated
op29 range baseline. The next direction is narrower: stress this under
many-calculator/op39 pressure, or replace hard assignment with a less
prescriptive credit signal.
