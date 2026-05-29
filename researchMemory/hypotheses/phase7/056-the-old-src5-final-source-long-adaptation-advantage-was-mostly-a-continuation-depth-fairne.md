# The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-selected-source-continuation-fairness.md

Summary:

- Giving the handoff-probe-selected `src5` step-1100 lineage the same extra 800-step frozen-policy continuation lifted it from `0.7950` to `0.8800`, then 1600-step stable-policy adaptation reached `0.9425`, nearly matching the old final-source `0.9500`.

Questions:

- What did we learn about The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue?
- Has The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue been tested?
- Should we repeat The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue?
- What is the status of The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue?
- What follow-up is allowed for The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-selected-source-continuation-fairness.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src5` step-1100 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.

Next Allowed:

- Apply the fair continuation recipe to `src4` step-1200, or optimize source acquisition for early handoff and continuation slope.

Full Text:

```text
POSITIVE: The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue.
Conclusion: Giving the handoff-probe-selected `src5` step-1100 lineage the same extra 800-step frozen-policy continuation lifted it from `0.7950` to `0.8800`, then 1600-step stable-policy adaptation reached `0.9425`, nearly matching the old final-source `0.9500`.
Do not repeat: Same `src5` step-1100 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Apply the fair continuation recipe to `src4` step-1200, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-selected-source-continuation-fairness.md`
```
