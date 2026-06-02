# 2026-06-02 Route-Heldout Shared Prior Review

## Why This Review Exists

The capped-prior many-calculator accounting showed the current best family
does not satisfy scalability: prompt-memory target discovery and prior fitting
remain linear per independent calculator. The next strategic question is
whether target-prior learning can be shared across routed calculators.

## What Changed

Added a route-heldout split to `scripts/diagnose_amortized_prior_from_trace.py`.
Instead of fitting on the normal prompt train split and evaluating prompt
heldout, it can now fit on all train-trace rows except specified
`calculator_hook_route` ids, then evaluate the prior on the withheld route.

Using the op29 capped-prior source trace, h128 numeric priors trained on three
of four `left_operand_mod` routes generalized to the withheld route:

| Heldout route | Fit rows | Heldout rows | Numeric heldout accuracy |
| ---: | ---: | ---: | ---: |
| `0` | `510` | `210` | `0.9333` |
| `1` | `499` | `221` | `0.9683` |
| `2` | `575` | `145` | `0.9793` |
| `3` | `576` | `144` | `0.9583` |

The route-0 embedding-prior control fit the three train routes perfectly
(`1.0000`) but got `0.0000` on the heldout route. That makes the positive
diagnostic specific to structured numeric sharing, not arbitrary memorization.

## Interpretation

This is the first positive result aimed directly at breaking per-calculator
target/prior scaling in family 14. It suggests that for homogeneous routed
calculators, a single structured prior can learn the target function from some
routes and supply targets to unscored routes.

But it is not yet the thesis result. It is post-hoc, uses true-discovered
targets from a fully trained capped source trace, and has not shown the source
policy can train when some routes do not receive sparse candidate scoring.

## What Should Stop

- More route-heldout diagnostic ladders as novelty. All four routes passed with
  numeric features, and the embedding control established the relevant failure
  mode.
- More cap/fraction/window tuning as a substitute for shared target learning.
- Treating this as proof of many-calculator training; it is only a target-prior
  feasibility diagnostic.

## What Deserves Compute

Train a routed source with sparse target discovery disabled or heavily reduced
for one or more routes, while a shared/global numeric prior supplies replay
targets across all routes. The gate should be the same as the current benchmark:
heldout source quality plus trusted 600-step frozen-policy additive handoff
with low controls.

## Are We Closer?

Yes, modestly. The many-calculator bottleneck is now sharper: structured shared
priors can generalize across homogeneous routes, so the next failure to test is
whether shared-prior replay can drive calculator-policy learning online without
per-route target discovery.
