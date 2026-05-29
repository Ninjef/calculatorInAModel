# A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-short-slope-selector-validation.md

Summary:

- On known `src2/src4/src5` handoff comparisons, 100-step loss/loss-drop selects the wrong checkpoint for `src5` and `src4`, though it selects `src2` final correctly; existing exact-match traces show `src5` still needs about 500 steps to select its known winner.

Questions:

- What did we learn about A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector?
- Has A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector been tested?
- Should we repeat A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector?
- What is the status of A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector?
- Why did A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-short-slope-selector-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same 0/25/50/100-step loss-slope probe over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.

Next Allowed:

- Keep 500/600-step handoff gates, optimize source acquisition against actual early handoff exact, or train a learned proxy on accumulated handoff traces.

Full Text:

```text
DISPROVEN: A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector.
Conclusion: On known `src2/src4/src5` handoff comparisons, 100-step loss/loss-drop selects the wrong checkpoint for `src5` and `src4`, though it selects `src2` final correctly; existing exact-match traces show `src5` still needs about 500 steps to select its known winner.
Do not repeat: Same 0/25/50/100-step loss-slope probe over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Keep 500/600-step handoff gates, optimize source acquisition against actual early handoff exact, or train a learned proxy on accumulated handoff traces.
Source: `aiAgentWorkHistory/phase7/2026-05-29-short-slope-selector-validation.md`
```
