# Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-route-weighted-prior-replay-source.md

Summary:

- Added `--result-boundary-target-amortized-prior-route-replay-routes` and `--result-boundary-target-amortized-prior-route-replay-weight`, which samples an extra prior-replay objective from selected routed hooks without adding candidate scoring or prompt-memory updates for those routes. A smoke test confirmed the route objective fired. On the full op19 route-excluded source, adding route 1 replay weight `2.0` improved final eval from `0.7875` to `327/400 = 0.8175`, train from `0.8406` to `0.8563`, and heldout from `0.5625` to `0.5750`, with causal final controls (`0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random). But best snapshot stayed `0.8075`, excluded route 1 stayed essentially unchanged (`0.7391` final snapshot and heldout route, `0.8000` diagnostic), prior train/heldout stayed weak (`0.7750`/`0.5750`), and forced evals rose to `58,800`. No handoff was run.

Questions:

- What did we learn about Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck?
- Has Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck been tested?
- Should we repeat Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck?
- What is the status of Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck?
- Why did Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-route-weighted-prior-replay-source.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run route-replay weight ladders, more op19 route-excluded repeats, op9 preflights, or route-heldout diagnostic ladders as novelty.

Next Allowed:

- Change the target/prior mechanism itself: explicit global/shared target discovery, route-shared prior training on candidate evidence before hard memory freezes, or a less-prescriptive credit signal that removes per-route prompt-memory tables and answer-derived candidate scoring.

Full Text:

```text
MIXED-NEGATIVE: Extra route-weighted prior replay only marginally improves the op19 route-excluded source and does not fix the shared-prior bottleneck.
Conclusion: Added `--result-boundary-target-amortized-prior-route-replay-routes` and `--result-boundary-target-amortized-prior-route-replay-weight`, which samples an extra prior-replay objective from selected routed hooks without adding candidate scoring or prompt-memory updates for those routes. A smoke test confirmed the route objective fired. On the full op19 route-excluded source, adding route 1 replay weight `2.0` improved final eval from `0.7875` to `327/400 = 0.8175`, train from `0.8406` to `0.8563`, and heldout from `0.5625` to `0.5750`, with causal final controls (`0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random). But best snapshot stayed `0.8075`, excluded route 1 stayed essentially unchanged (`0.7391` final snapshot and heldout route, `0.8000` diagnostic), prior train/heldout stayed weak (`0.7750`/`0.5750`), and forced evals rose to `58,800`. No handoff was run.
Do not repeat: Do not run route-replay weight ladders, more op19 route-excluded repeats, op9 preflights, or route-heldout diagnostic ladders as novelty.
Next allowed test: Change the target/prior mechanism itself: explicit global/shared target discovery, route-shared prior training on candidate evidence before hard memory freezes, or a less-prescriptive credit signal that removes per-route prompt-memory tables and answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-02-route-weighted-prior-replay-source.md`
```
