# Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md

Summary:

- Added `--result-policy-improvement-assignment-refresh-interval` to refresh exact full-result hard-assignment targets every N steps on fixed exhaustive-grid batches and reuse cached targets between refreshes. Against the same op19 `rhead64` exact ceiling, refresh2 should cut assignment scoring calls from `201` refreshes to `101`, and refresh5 to `41`. But source quality fell well below exact: exact reached best snapshot `0.8625` and final `0.7350`; refresh2 reached best snapshot `0.5875` and final `237/400 = 0.5925`; refresh5 reached best snapshot/final `0.4950`. Target accuracy at step `200` remained decent for refresh2 (`0.9603`) but the stale-target cadence slowed source acquisition, and local wall time barely improved in the full diagnostic gate (`115.5s` exact, `106.4s` refresh2, `105.1s` refresh5) because snapshots/checkpoints/other objectives dominate.

Questions:

- What did we learn about Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal?
- Has Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal been tested?
- Should we repeat Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal?
- What is the status of Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal?
- Why did Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more fixed refresh-interval ladders on the same op19 `rhead64` 200-step gate as novelty. Fixed stale exact targets are not a good assignment-cost reduction path at this budget.

Next Allowed:

- Temporal amortization needs an adaptive freshness/trust criterion, predictive target update, or other mechanism that preserves exact-ceiling source acquisition while proving real wall-clock or many-calculator savings. Otherwise prioritize different credit-assignment mechanisms.

Full Text:

```text
MIXED-NEGATIVE: Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal.
Conclusion: Added `--result-policy-improvement-assignment-refresh-interval` to refresh exact full-result hard-assignment targets every N steps on fixed exhaustive-grid batches and reuse cached targets between refreshes. Against the same op19 `rhead64` exact ceiling, refresh2 should cut assignment scoring calls from `201` refreshes to `101`, and refresh5 to `41`. But source quality fell well below exact: exact reached best snapshot `0.8625` and final `0.7350`; refresh2 reached best snapshot `0.5875` and final `237/400 = 0.5925`; refresh5 reached best snapshot/final `0.4950`. Target accuracy at step `200` remained decent for refresh2 (`0.9603`) but the stale-target cadence slowed source acquisition, and local wall time barely improved in the full diagnostic gate (`115.5s` exact, `106.4s` refresh2, `105.1s` refresh5) because snapshots/checkpoints/other objectives dominate.
Do not repeat: Do not run more fixed refresh-interval ladders on the same op19 `rhead64` 200-step gate as novelty. Fixed stale exact targets are not a good assignment-cost reduction path at this budget.
Next allowed test: Temporal amortization needs an adaptive freshness/trust criterion, predictive target update, or other mechanism that preserves exact-ceiling source acquisition while proving real wall-clock or many-calculator savings. Otherwise prioritize different credit-assignment mechanisms.
Source: `aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md`
```
