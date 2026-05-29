# Fair continuation also improves the weak `src4` selected-source lineage.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-src4-selected-source-continuation-fairness.md

Summary:

- `src4` step-1200 selected handoff improved from `0.7800` to `0.8150` with another 800 frozen-policy steps, then reached `0.9125` after policy-backbone-frozen long adaptation, beating direct selected long (`0.8900`) and old final-source long (`0.7550`).

Questions:

- What did we learn about Fair continuation also improves the weak `src4` selected-source lineage?
- Has Fair continuation also improves the weak `src4` selected-source lineage been tested?
- Should we repeat Fair continuation also improves the weak `src4` selected-source lineage?
- What is the status of Fair continuation also improves the weak `src4` selected-source lineage?
- What follow-up is allowed for Fair continuation also improves the weak `src4` selected-source lineage?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-src4-selected-source-continuation-fairness.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src4` step-1200 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.

Next Allowed:

- Reduce handoff-probe/continuation cost, or optimize source acquisition for early handoff and continuation slope.

Full Text:

```text
POSITIVE: Fair continuation also improves the weak `src4` selected-source lineage.
Conclusion: `src4` step-1200 selected handoff improved from `0.7800` to `0.8150` with another 800 frozen-policy steps, then reached `0.9125` after policy-backbone-frozen long adaptation, beating direct selected long (`0.8900`) and old final-source long (`0.7550`).
Do not repeat: Same `src4` step-1200 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Reduce handoff-probe/continuation cost, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src4-selected-source-continuation-fairness.md`
```
