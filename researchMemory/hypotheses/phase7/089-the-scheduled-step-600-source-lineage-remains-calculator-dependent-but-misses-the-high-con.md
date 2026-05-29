# The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md

Summary:

- Starting from scheduled step-600 handoff final `0.7725`, 800-step frozen-policy continuation reached only `0.7775`, 600-step readout after continuation reached `0.8175`, and an extra 1000 stable-policy readout steps reached `0.8475`; controls stayed low (`injection_zero <=0.0547`, forced-random <=`0.0391`), but learned calc stayed around `0.5391`.

Questions:

- What did we learn about The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate?
- Has The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate been tested?
- Should we repeat The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate?
- What is the status of The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate?
- What follow-up is allowed for The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same scheduled step-600 handoff -> 800 continuation -> 600 readout -> extra 1000 readout chain as novelty.

Next Allowed:

- Improve source policy accuracy while preserving scheduled geometry, or run continuation/readout only after a scheduled source checkpoint shows both strong handoff geometry and materially higher learned calculator accuracy.

Full Text:

```text
MIXED-POSITIVE: The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate.
Conclusion: Starting from scheduled step-600 handoff final `0.7725`, 800-step frozen-policy continuation reached only `0.7775`, 600-step readout after continuation reached `0.8175`, and an extra 1000 stable-policy readout steps reached `0.8475`; controls stayed low (`injection_zero <=0.0547`, forced-random <=`0.0391`), but learned calc stayed around `0.5391`.
Do not repeat: The same scheduled step-600 handoff -> 800 continuation -> 600 readout -> extra 1000 readout chain as novelty.
Next allowed test: Improve source policy accuracy while preserving scheduled geometry, or run continuation/readout only after a scheduled source checkpoint shows both strong handoff geometry and materially higher learned calculator accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`
```
