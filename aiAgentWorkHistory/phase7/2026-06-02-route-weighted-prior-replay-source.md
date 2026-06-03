# Route-Weighted Prior Replay Source

Date: 2026-06-02

## Goal

Test whether the partial op19 route-excluded shared-prior source failed because
the excluded route received too little replay pressure from the shared numeric
prior.

## Implementation

Added two CLI flags:

- `--result-boundary-target-amortized-prior-route-replay-routes`
- `--result-boundary-target-amortized-prior-route-replay-weight`

When enabled, source training builds a route-filtered replay pool from the
global train/heldout prompt pools and adds an extra prior-pseudo-target loss on
the selected routes. This does not add candidate scoring or prompt-memory target
updates for those routes.

Verification:

- `python3 -m py_compile scripts/overfit_one_batch.py`
- `python3 -m pytest tests/test_model.py -k "subset_arithmetic_batch_by_routes or prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior or route_exclusion"`
  -> `6 passed, 151 deselected`
- Smoke run:
  `runs/2026-06-02_route_replay_smoke/.../model-c-2digit-seed5`
  recorded `result_boundary_target_amortized_prior_route_replay_objective=7.2753`
  and route replay pool count `3` in the training curve.

## Full Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_routereplay1w2_src5000/2026-06-02_175525_793893_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-1b6e8a4258/model-c-2digit-seed9
```

Single experimental delta from the prior op19 route-excluded source:

```text
--result-boundary-target-amortized-prior-route-replay-routes 1
--result-boundary-target-amortized-prior-route-replay-weight 2.0
```

## Results

| Metric | Value |
| --- | ---: |
| Final eval exact / calculator-result accuracy | `327/400 = 0.8175` |
| Best snapshot normal / calculator-result accuracy | `0.8075` |
| Snapshot injection-zero / forced-zero / forced-random | `0.0475 / 0.0025 / 0.0025` |
| 128-sample diagnostic exact / calculator-result accuracy | `0.84375 / 0.84375` |
| 128-sample diagnostic injection-zero / forced-zero / forced-random | `0.0703125 / 0.0078125 / 0.015625` |
| Train prompt exact / calculator-result accuracy | `0.85625` |
| Heldout prompt exact / calculator-result accuracy | `0.5750` |
| Prompt-memory entries / expected | `223 / 223` |
| Forced-result evals | `58,800` |
| Prior updates | `2,501` |
| Prior train / heldout accuracy | `0.7750 / 0.5750` |
| Route replay pool count | `120` |
| Snapshot route 1 exact / calculator-result accuracy | `0.7391 / 0.7391` |
| 128-sample diagnostic route 1 exact / calculator-result accuracy | `0.8000 / 0.8000` |
| Heldout route 1 exact / calculator-result accuracy | `0.7391 / 0.7391` |

## Interpretation

This is mixed-negative for route-weighted replay as the missing mechanism. It
improves final eval modestly over the previous route-excluded source (`0.8175`
vs `0.7875`) and stays causal, but it does not improve the best snapshot,
excluded route, or prior enough to justify trusted handoff.

The result says the problem is not merely weak replay pressure on the excluded
route. The shared prior itself is still too weak/noisy under this hard-memory
target discovery process.

## Do Not Repeat

- Do not run route-replay weight ladders as novelty.
- Do not rerun the same op19 route-excluded source or op9 preflights.
- Do not treat route-heldout diagnostics as a substitute for source gates.

## Next

Change how shared targets are learned: explicit global/shared target discovery,
route-shared prior training on candidate evidence before hard memory freezes, or
a less-prescriptive credit signal that removes per-route prompt-memory tables and
answer-derived candidate scoring.
