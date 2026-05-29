# Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-reduced-continuation-budget-validation.md

Summary:

- With 600-step readout after reduced continuation, `src5` still passed (`0.9275` vs `0.9325` reference), but weak `src4` fell below gate (`0.8750` vs `0.9025` reference) despite retained calculator dependence.

Questions:

- What did we learn about Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive?
- Has Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive been tested?
- Should we repeat Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive?
- What is the status of Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive?
- Why did Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-reduced-continuation-budget-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same 600-step continuation plus 600-step policy-backbone-frozen readout from selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.

Next Allowed:

- Keep 800 continuation for weak sources, test 700-step continuation only for fine-grained tuning, or optimize source acquisition for continuation slope.

Full Text:

```text
MIXED-NEGATIVE: Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive.
Conclusion: With 600-step readout after reduced continuation, `src5` still passed (`0.9275` vs `0.9325` reference), but weak `src4` fell below gate (`0.8750` vs `0.9025` reference) despite retained calculator dependence.
Do not repeat: Same 600-step continuation plus 600-step policy-backbone-frozen readout from selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Keep 800 continuation for weak sources, test 700-step continuation only for fine-grained tuning, or optimize source acquisition for continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-continuation-budget-validation.md`
```
