# Op19 Route-Excluded Shared-Prior Source

Date: 2026-06-02

## Goal

Test the full source gate recommended after the route-heldout diagnostic and
corrected op9 preflight: can a shared numeric prior train a routed calculator
whose prompt-memory target discovery is disabled?

## Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_src5000_fixed/2026-06-02_172119_098782_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-1d610d942a/model-c-2digit-seed9
```

Key settings:

- `operand_max=19`, exhaustive grid, batch64 streaming train with 20% heldout.
- Four `left_operand_mod` routed hooks with shared calculator output projection.
- Prompt-keyed online hard memory, frozen when full.
- `--result-boundary-target-memory-update-exclude-routes 1`.
- Numeric amortized prior, full-memory fit every 2 steps, train-accuracy stop at
  `1.0` with patience `100`.

## Results

| Metric | Value |
| --- | ---: |
| Final eval exact / calculator-result accuracy | `315/400 = 0.7875` |
| Best snapshot normal / calculator-result accuracy | `0.8075` |
| Snapshot injection-zero / forced-zero / forced-random | `0.0475 / 0.0025 / 0.0025` |
| 128-sample diagnostic exact / calculator-result accuracy | `0.8125 / 0.8125` |
| 128-sample diagnostic injection-zero / forced-zero / forced-random | `0.0703125 / 0.0078125 / 0.015625` |
| Train prompt exact / calculator-result accuracy | `0.840625` |
| Heldout prompt exact / calculator-result accuracy | `0.5625` |
| Train prompt injection-zero / forced-zero / forced-random | `0.046875 / 0.003125 / 0.01875` |
| Heldout prompt injection-zero / forced-zero / forced-random | `0.0500 / 0.0000 / 0.0125` |
| Prompt-memory entries / expected | `223 / 223` |
| Last score-eligible / update-excluded fraction | `0.6875 / 0.3125` |
| Forced-result evals | `37,896` |
| Prior updates | `2,501` |
| Prior train / heldout accuracy | `0.7781 / 0.5625` |
| Snapshot route 1 exact / calculator-result accuracy | `0.7304 / 0.7304` |
| 128-sample diagnostic route 1 exact / calculator-result accuracy | `0.8000 / 0.8000` |
| Heldout route 1 exact / calculator-result accuracy | `0.7391 / 0.7391` |

## Interpretation

This is mixed-positive. Route 1, which was excluded from direct prompt-memory
updates, was not dead: it learned causally from shared numeric-prior replay. That
is a stronger signal than the corrected op9 preflight and supports the
route-heldout diagnostic's claim that numeric target structure can share across
routed calculators.

It does not clear the source gate. Overall source quality, heldout prompt
quality, and online prior accuracy are too low for trusted handoff, so no
handoff was run.

## Do Not Repeat

- Do not rerun this exact op19 route-excluded 5000-step recipe as novelty.
- Do not run more short op9 route-exclusion preflights.
- Do not run route-heldout diagnostic ladders; the diagnostic already showed
  numeric sharing and embedding memorization failure.
- Do not run cadence or patience variants unless the objective changes.

## Next

Change the mechanism rather than the schedule: explicit shared/global prior
targets, route-balanced/global replay, shared target discovery across
calculators, or a less-prescriptive credit mechanism that removes per-route
prompt-memory tables and answer-derived candidate scoring.
