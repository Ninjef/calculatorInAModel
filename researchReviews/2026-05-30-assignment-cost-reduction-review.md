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

## What Deserves Compute

- Coverage-aware or active candidate construction that explicitly raises
  true/best-result inclusion without scoring the full vocabulary.
- Structured proposals that exploit arithmetic/result geometry while still
  being validated against exact-grid assignment ceilings.
- Non-enumerative credit signals that avoid hard assignment rather than
  sampling it thinly.
- Adaptive target-refresh criteria that can skip scoring only when target
  freshness is likely preserved, and that report real compute savings.
- Many-calculator accounting, but only when paired with a candidate mechanism
  whose single-calculator ceiling comparison is not already negative.

## Are We Closer?

Slightly. This did not solve scalable assignment, but it closed a tempting cheap
path quickly and instrumented the code so future assignment approximations can
report coverage and target-quality diagnostics. The next direction is narrower:
reduce assignment cost only with a proposal/credit mechanism that preserves the
exact source signal, not with duplicate-prone uniform sampling.
