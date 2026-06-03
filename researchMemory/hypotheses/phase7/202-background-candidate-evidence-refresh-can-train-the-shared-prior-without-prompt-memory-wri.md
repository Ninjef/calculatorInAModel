# Background candidate-evidence refresh can train the shared prior without prompt-memory writes.

Kind: hypothesis_memory
Status: TOOLING
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-background-candidate-evidence-refresh-tooling.md

Summary:

- Added `--result-boundary-target-amortized-prior-evidence-refresh-*` flags so the run can periodically score fresh train-pool prompts and update the shared amortized prior without adding prompt-memory entries. The refresh can exclude routed hooks from evidence scoring while prior replay still trains those routes. A tiny route-excluded smoke with route `1` excluded recorded refresh updates/examples `2/5` and refresh forced evals `76`, with final metrics and training-curve refresh fields populated.

Questions:

- What did we learn about Background candidate-evidence refresh can train the shared prior without prompt-memory writes?
- Has Background candidate-evidence refresh can train the shared prior without prompt-memory writes been tested?
- Should we repeat Background candidate-evidence refresh can train the shared prior without prompt-memory writes?
- What is the status of Background candidate-evidence refresh can train the shared prior without prompt-memory writes?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-background-candidate-evidence-refresh-tooling.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat the tiny smoke as source-quality evidence, and do not use this to reopen route-weight/bootstrap/candidate-evidence timing ladders.

Next Allowed:

- Run a real op19 route-excluded source where background evidence refresh scores only non-excluded routes and prior replay trains all routes, then require heldout/excluded-route quality before any trusted handoff.

Full Text:

```text
TOOLING: Background candidate-evidence refresh can train the shared prior without prompt-memory writes.
Conclusion: Added `--result-boundary-target-amortized-prior-evidence-refresh-*` flags so the run can periodically score fresh train-pool prompts and update the shared amortized prior without adding prompt-memory entries. The refresh can exclude routed hooks from evidence scoring while prior replay still trains those routes. A tiny route-excluded smoke with route `1` excluded recorded refresh updates/examples `2/5` and refresh forced evals `76`, with final metrics and training-curve refresh fields populated.
Do not repeat: Do not treat the tiny smoke as source-quality evidence, and do not use this to reopen route-weight/bootstrap/candidate-evidence timing ladders.
Next allowed test: Run a real op19 route-excluded source where background evidence refresh scores only non-excluded routes and prior replay trains all routes, then require heldout/excluded-route quality before any trusted handoff.
Source: `aiAgentWorkHistory/phase7/2026-06-03-background-candidate-evidence-refresh-tooling.md`
```
