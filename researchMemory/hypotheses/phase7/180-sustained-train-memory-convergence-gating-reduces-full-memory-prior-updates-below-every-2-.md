# Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-convergence-stop-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-stop-train-accuracy` and `--result-boundary-target-amortized-prior-stop-patience` so fitting can stop after prompt memory is full and the prior has stayed converged. Stopping at the first train-memory `1.0` cut updates to `1029` but hurt heldout source (`0.875` heldout, `0.9825` overall), showing train fit alone is too optimistic. Requiring `100` converged fit updates preserved the every-2 source gate with fewer updates: overall `398/400 = 0.9950`, train `1.0000`, heldout `0.9125`, low heldout controls (`0.0500/0.0000/0.0125`), prior train/heldout `1.0000/0.9125`, forced evals `86,016`, and prior updates `1889` instead of `2501`/`5001`. The trusted frozen-policy additive handoff reached `397/400 = 0.9925`, diagnostic calc `0.984375`, and low 128-sample controls (`0.0546875` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).

Questions:

- What did we learn about Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality?
- Has Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality been tested?
- Should we repeat Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality?
- What is the status of Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality?
- What follow-up is allowed for Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-convergence-stop-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run patience ladders as novelty. First-hit convergence is disproven; patience-100 is the new safe train-convergence benchmark.

Next Allowed:

- Use a validation/heldout-prior signal or coreset/reservoir prior batches to reduce below `1889` updates while preserving the same heldout source and trusted handoff gate.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality.
Conclusion: Added `--result-boundary-target-amortized-prior-stop-train-accuracy` and `--result-boundary-target-amortized-prior-stop-patience` so fitting can stop after prompt memory is full and the prior has stayed converged. Stopping at the first train-memory `1.0` cut updates to `1029` but hurt heldout source (`0.875` heldout, `0.9825` overall), showing train fit alone is too optimistic. Requiring `100` converged fit updates preserved the every-2 source gate with fewer updates: overall `398/400 = 0.9950`, train `1.0000`, heldout `0.9125`, low heldout controls (`0.0500/0.0000/0.0125`), prior train/heldout `1.0000/0.9125`, forced evals `86,016`, and prior updates `1889` instead of `2501`/`5001`. The trusted frozen-policy additive handoff reached `397/400 = 0.9925`, diagnostic calc `0.984375`, and low 128-sample controls (`0.0546875` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).
Do not repeat: Do not run patience ladders as novelty. First-hit convergence is disproven; patience-100 is the new safe train-convergence benchmark.
Next allowed test: Use a validation/heldout-prior signal or coreset/reservoir prior batches to reduce below `1889` updates while preserving the same heldout source and trusted handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-convergence-stop-gate.md`
```
