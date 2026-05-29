# The 500-step selector does not generalize cleanly to fresh `src6`.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md

Summary:

- Fresh `src6` 500-step handoff scores pick step `1500` (`0.7200`) over final (`0.6850`), but full 800-step handoff is better from final (`0.8975` vs `0.8875`); 600-step scores would pick final (`0.8050` vs `0.7800`).

Questions:

- What did we learn about The 500-step selector does not generalize cleanly to fresh `src6`?
- Has The 500-step selector does not generalize cleanly to fresh `src6` been tested?
- Should we repeat The 500-step selector does not generalize cleanly to fresh `src6`?
- What is the status of The 500-step selector does not generalize cleanly to fresh `src6`?
- Why did The 500-step selector does not generalize cleanly to fresh `src6` fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src6` step-1200/step-1500/final additive seed-6 frozen-policy 800-step comparison as novelty.

Next Allowed:

- Use 600-step selection for fresh sources, run continuation/readout from fresh `src6` final, or optimize source acquisition for 600-step handoff/continuation slope.

Full Text:

```text
MIXED-NEGATIVE: The 500-step selector does not generalize cleanly to fresh `src6`.
Conclusion: Fresh `src6` 500-step handoff scores pick step `1500` (`0.7200`) over final (`0.6850`), but full 800-step handoff is better from final (`0.8975` vs `0.8875`); 600-step scores would pick final (`0.8050` vs `0.7800`).
Do not repeat: Same `src6` step-1200/step-1500/final additive seed-6 frozen-policy 800-step comparison as novelty.
Next allowed test: Use 600-step selection for fresh sources, run continuation/readout from fresh `src6` final, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`
```
