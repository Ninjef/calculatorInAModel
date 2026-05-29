# A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md

Summary:

- Leave-family-out ridge over 21 deduped candidates and 8 source families reached `3/8`, `4/8`, `3/8`, and `5/8` winner accuracy at prediction steps `200/300/400/500`; raw early exact matched or beat it at every step and reached `6/8` at step `500`.

Questions:

- What did we learn about A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate?
- Has A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate been tested?
- Should we repeat A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate?
- What is the status of A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate?
- Why did A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same ridge selector over normal/zero/oracle/forced-random/calc early trace features on the current Phase 7 handoff trace dataset as novelty.

Next Allowed:

- Add logging-only in-training additive handoff probes, collect more labeled families, or test a richer learned selector only if it beats raw early exact under leave-family-out validation.

Full Text:

```text
DISPROVEN: A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate.
Conclusion: Leave-family-out ridge over 21 deduped candidates and 8 source families reached `3/8`, `4/8`, `3/8`, and `5/8` winner accuracy at prediction steps `200/300/400/500`; raw early exact matched or beat it at every step and reached `6/8` at step `500`.
Do not repeat: Same ridge selector over normal/zero/oracle/forced-random/calc early trace features on the current Phase 7 handoff trace dataset as novelty.
Next allowed test: Add logging-only in-training additive handoff probes, collect more labeled families, or test a richer learned selector only if it beats raw early exact under leave-family-out validation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`
```
