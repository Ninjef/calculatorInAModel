# A 600-step handoff probe can select a better source checkpoint than source accuracy.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md

Summary:

- On `src5`, the 600-step probe selected step `1100` (source normal `0.8400`) over the source-accuracy-selected step `1500` (`0.9200`); full additive handoff improved from `0.6975` to `0.7950`.

Questions:

- What did we learn about A 600-step handoff probe can select a better source checkpoint than source accuracy?
- Has A 600-step handoff probe can select a better source checkpoint than source accuracy been tested?
- Should we repeat A 600-step handoff probe can select a better source checkpoint than source accuracy?
- What is the status of A 600-step handoff probe can select a better source checkpoint than source accuracy?
- What follow-up is allowed for A 600-step handoff probe can select a better source checkpoint than source accuracy?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src5` step `1100/1400/1500/final`, additive seed `5`, frozen-policy handoff-probe comparison as novelty.

Next Allowed:

- Use the 600-step handoff probe on newly acquired source checkpoints, reduce/approximate its cost, or optimize source acquisition for probe score.

Full Text:

```text
POSITIVE: A 600-step handoff probe can select a better source checkpoint than source accuracy.
Conclusion: On `src5`, the 600-step probe selected step `1100` (source normal `0.8400`) over the source-accuracy-selected step `1500` (`0.9200`); full additive handoff improved from `0.6975` to `0.7950`.
Do not repeat: Same `src5` step `1100/1400/1500/final`, additive seed `5`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use the 600-step handoff probe on newly acquired source checkpoints, reduce/approximate its cost, or optimize source acquisition for probe score.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`
```
