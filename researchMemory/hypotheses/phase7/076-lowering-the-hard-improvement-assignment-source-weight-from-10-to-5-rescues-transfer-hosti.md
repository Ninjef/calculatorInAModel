# Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md

Summary:

- The weight-5 seed-10 source weakened to final eval `0.6750`; its best source snapshots were around `0.78`, and 600-step additive handoffs from step `1200`/final reached only `0.3425`/`0.2475` snapshots and `0.3000`/`0.2325` final eval.

Questions:

- What did we learn about Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry?
- Has Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry been tested?
- Should we repeat Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry?
- What is the status of Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry?
- Why did Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same seed-10 no-decay entropy `0.05`, diversity `0.1`, improvement weight `5`, 1600-step source run or step-1200/final 600-step frozen handoffs as novelty.

Next Allowed:

- Optimize source acquisition against actual 500/600-step handoff behavior, add a direct handoff/readout geometry term, or train a learned selector validated against the handoff gate.

Full Text:

```text
DISPROVEN: Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry.
Conclusion: The weight-5 seed-10 source weakened to final eval `0.6750`; its best source snapshots were around `0.78`, and 600-step additive handoffs from step `1200`/final reached only `0.3425`/`0.2475` snapshots and `0.3000`/`0.2325` final eval.
Do not repeat: Same seed-10 no-decay entropy `0.05`, diversity `0.1`, improvement weight `5`, 1600-step source run or step-1200/final 600-step frozen handoffs as novelty.
Next allowed test: Optimize source acquisition against actual 500/600-step handoff behavior, add a direct handoff/readout geometry term, or train a learned selector validated against the handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`
```
