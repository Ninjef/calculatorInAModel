# Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-fit-cadence-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-fit-every` to decouple prior-fit cadence from model replay. On the same op19 four-hook shared-output heldout source, fitting every `10` steps cut prior updates to `501` but underfit the prior (train `0.953125`, heldout prior `0.7875`) and degraded the source to overall `0.9475`, train `0.978125`, heldout `0.7625`. Fitting every `2` steps cut prior updates from `5001` to `2501` while preserving the benchmark source: overall `0.9950`, train `1.0000`, heldout `0.9125`, heldout controls `0.0500/0.0000/0.0125`, prior train/heldout `1.0000/0.9125`, and forced evals still `86,016`. The trusted frozen-policy additive handoff from the every-2 source reached final `395/400 = 0.9875`, diagnostic calc `0.984375`, and low 128-sample controls (`0.015625` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).

Questions:

- What did we learn about Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits?
- Has Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits been tested?
- Should we repeat Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits?
- What is the status of Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits?
- What follow-up is allowed for Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-fit-cadence-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run a cadence ladder (`3/4/5/8/10`) as novelty. Every-10 already identifies prior update starvation; every-2 is the safe benchmark.

Next Allowed:

- Replace cadence-only thinning with convergence-gated fitting, stop/refresh after memory/prior convergence, or coreset/reservoir prior batches that target fewer than `2501` full-memory updates while preserving the every-2 heldout/handoff gate.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits.
Conclusion: Added `--result-boundary-target-amortized-prior-fit-every` to decouple prior-fit cadence from model replay. On the same op19 four-hook shared-output heldout source, fitting every `10` steps cut prior updates to `501` but underfit the prior (train `0.953125`, heldout prior `0.7875`) and degraded the source to overall `0.9475`, train `0.978125`, heldout `0.7625`. Fitting every `2` steps cut prior updates from `5001` to `2501` while preserving the benchmark source: overall `0.9950`, train `1.0000`, heldout `0.9125`, heldout controls `0.0500/0.0000/0.0125`, prior train/heldout `1.0000/0.9125`, and forced evals still `86,016`. The trusted frozen-policy additive handoff from the every-2 source reached final `395/400 = 0.9875`, diagnostic calc `0.984375`, and low 128-sample controls (`0.015625` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).
Do not repeat: Do not run a cadence ladder (`3/4/5/8/10`) as novelty. Every-10 already identifies prior update starvation; every-2 is the safe benchmark.
Next allowed test: Replace cadence-only thinning with convergence-gated fitting, stop/refresh after memory/prior convergence, or coreset/reservoir prior batches that target fewer than `2501` full-memory updates while preserving the every-2 heldout/handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-fit-cadence-gate.md`
```
