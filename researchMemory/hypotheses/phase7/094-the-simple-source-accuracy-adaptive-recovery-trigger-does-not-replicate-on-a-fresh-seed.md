# The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md

Summary:

- On a fresh seed-17 source run, the same `result_policy_argmax_result_accuracy >= 0.65` min-step-500 trigger never fired; source final eval reached only `0.6100`, and trusted 600-step frozen-policy handoff reached `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`. A matched fixed step-600 control did better but still missed the high gate: source final `0.7450`, handoff `0.7675` final / `0.7850` snapshot, learned calc `0.7350`, injection-zero `0.0500`, forced-random `0.0375`.

Questions:

- What did we learn about The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed?
- Has The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed been tested?
- Should we repeat The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed?
- What is the status of The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed?
- Why did The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same fresh seed-17 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive run or matched fixed step-600 control as novelty.

Next Allowed:

- Use a smoothed/conjunctive recovery trigger or a different transition metric; do not treat raw source argmax accuracy thresholding as validated.

Full Text:

```text
MIXED-NEGATIVE: The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed.
Conclusion: On a fresh seed-17 source run, the same `result_policy_argmax_result_accuracy >= 0.65` min-step-500 trigger never fired; source final eval reached only `0.6100`, and trusted 600-step frozen-policy handoff reached `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`. A matched fixed step-600 control did better but still missed the high gate: source final `0.7450`, handoff `0.7675` final / `0.7850` snapshot, learned calc `0.7350`, injection-zero `0.0500`, forced-random `0.0375`.
Do not repeat: The same fresh seed-17 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive run or matched fixed step-600 control as novelty.
Next allowed test: Use a smoothed/conjunctive recovery trigger or a different transition metric; do not treat raw source argmax accuracy thresholding as validated.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`
```
