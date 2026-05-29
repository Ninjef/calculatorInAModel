# Longer scheduled forced-true source acquisition compounds handoff geometry through step 600.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md

Summary:

- Extending the seed-13 scheduled source to `800` steps improved forced-result geometry from step `200` to `600` (`forced_best_true 0.2125 -> 0.9800`, 50-step slope loss `1.0360 -> 0.4719`) and the step-600 source checkpoint reached `0.7725` final eval under the trusted 600-step frozen-policy additive handoff. Step `800` had perfect forced-result geometry but worse handoff (`0.6750` final), so final source checkpoint is not automatically best.

Questions:

- What did we learn about Longer scheduled forced-true source acquisition compounds handoff geometry through step 600?
- Has Longer scheduled forced-true source acquisition compounds handoff geometry through step 600 been tested?
- Should we repeat Longer scheduled forced-true source acquisition compounds handoff geometry through step 600?
- What is the status of Longer scheduled forced-true source acquisition compounds handoff geometry through step 600?
- What follow-up is allowed for Longer scheduled forced-true source acquisition compounds handoff geometry through step 600?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-13 scheduled source `200/400/600/800` geometry ladder or step-600 vs step-800 handoff comparison as novelty.

Next Allowed:

- Run continuation/readout from the step-600 handoff lineage to test whether the scheduled source can clear the high non-bottleneck gate; replicate on a fresh seed only if stability is the explicit question.

Full Text:

```text
POSITIVE: Longer scheduled forced-true source acquisition compounds handoff geometry through step 600.
Conclusion: Extending the seed-13 scheduled source to `800` steps improved forced-result geometry from step `200` to `600` (`forced_best_true 0.2125 -> 0.9800`, 50-step slope loss `1.0360 -> 0.4719`) and the step-600 source checkpoint reached `0.7725` final eval under the trusted 600-step frozen-policy additive handoff. Step `800` had perfect forced-result geometry but worse handoff (`0.6750` final), so final source checkpoint is not automatically best.
Do not repeat: The same seed-13 scheduled source `200/400/600/800` geometry ladder or step-600 vs step-800 handoff comparison as novelty.
Next allowed test: Run continuation/readout from the step-600 handoff lineage to test whether the scheduled source can clear the high non-bottleneck gate; replicate on a fresh seed only if stability is the explicit question.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`
```
