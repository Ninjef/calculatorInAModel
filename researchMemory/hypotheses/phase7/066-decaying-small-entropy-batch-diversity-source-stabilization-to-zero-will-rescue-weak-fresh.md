# Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-variant.md

Summary:

- Fresh seed-9 source acquisition with entropy `0.05`, batch diversity `0.1`, improvement assignment `10`, and decay-to-zero over 1200 steps peaked at `0.7050` around steps `700/900`, then collapsed to final `0.1825`.

Questions:

- What did we learn about Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition?
- Has Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition been tested?
- Should we repeat Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition?
- What is the status of Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition?
- Why did Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-variant.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same decay-to-zero source-only recipe with answer loss off as novelty.

Next Allowed:

- Keep a nonzero source-objective floor, add policy anchoring, or optimize source acquisition for 600-step handoff/continuation slope.

Full Text:

```text
DISPROVEN: Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition.
Conclusion: Fresh seed-9 source acquisition with entropy `0.05`, batch diversity `0.1`, improvement assignment `10`, and decay-to-zero over 1200 steps peaked at `0.7050` around steps `700/900`, then collapsed to final `0.1825`.
Do not repeat: Same decay-to-zero source-only recipe with answer loss off as novelty.
Next allowed test: Keep a nonzero source-objective floor, add policy anchoring, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-variant.md`
```
