# The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-shorter-handoff-probe-trace-audit.md

Summary:

- Existing traces show 400-step probe would pick `src5` step-1500, but 500-step probe picks the same checkpoints as 600 for both audited families (`src5` step-1100, `src4` step-1200).

Questions:

- What did we learn about The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400?
- Has The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400 been tested?
- Should we repeat The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400?
- What is the status of The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400?
- What follow-up is allowed for The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-shorter-handoff-probe-trace-audit.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same trace audit over existing `src5` 1100/1400/1500/final and `src4` 1000/1200/final handoff probes as novelty.

Next Allowed:

- Validate 500-step selection on new source checkpoints, reduce the 800-step continuation cost, or optimize source acquisition for early handoff/continuation slope.

Full Text:

```text
MIXED-POSITIVE: The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400.
Conclusion: Existing traces show 400-step probe would pick `src5` step-1500, but 500-step probe picks the same checkpoints as 600 for both audited families (`src5` step-1100, `src4` step-1200).
Do not repeat: Same trace audit over existing `src5` 1100/1400/1500/final and `src4` 1000/1200/final handoff probes as novelty.
Next allowed test: Validate 500-step selection on new source checkpoints, reduce the 800-step continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-shorter-handoff-probe-trace-audit.md`
```
