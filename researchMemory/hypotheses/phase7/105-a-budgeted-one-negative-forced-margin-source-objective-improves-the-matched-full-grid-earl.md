# A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-op19-gate.md

Summary:

- The full-grid 4-negative forced-margin branch was too costly locally and was stopped after it only wrote config. Reducing to one sampled negative per prompt made the contrastive objective practical and positive on the matched `operand_max=19`, seed-13, 200-step source gate. The source reached `0.3225` train calc / `0.3600` final eval, above the earlier scheduled forced-true 200-step source (`0.2800`/`0.2750`). Geometry was mixed: forced-result ranking was strong (`forced_best_true=0.6725`) but 50-step slope final loss was `1.4660`, worse than scheduled forced-true (`1.0360`). The trusted 600-step frozen-policy additive handoff resolved the conflict positively: final eval `0.6600`, step-600 normal `0.7050`, injection-zero `0.0000`, forced-random `0.0250`, learned calc `0.3375`, beating the matched scheduled forced-true handoff (`0.4150`) and baseline (`0.2525`).

Questions:

- What did we learn about A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate?
- Has A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate been tested?
- Should we repeat A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate?
- What is the status of A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate?
- Why did A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate fail?
- What follow-up is allowed for A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-op19-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same seed-13, `operand_max=19`, 200-step one-negative forced-margin source plus 600-step handoff as novelty, and do not run the 4-negative full-grid branch without a compute-reduction plan.

Next Allowed:

- Extend the one-negative forced-margin source to longer horizons (`400/600`) and verify with trusted 600-step handoff, or replicate on a fresh seed if the explicit question is stability. Keep slope/geometry as diagnostics only; actual handoff remains arbiter.

Full Text:

```text
POSITIVE: A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate.
Conclusion: The full-grid 4-negative forced-margin branch was too costly locally and was stopped after it only wrote config. Reducing to one sampled negative per prompt made the contrastive objective practical and positive on the matched `operand_max=19`, seed-13, 200-step source gate. The source reached `0.3225` train calc / `0.3600` final eval, above the earlier scheduled forced-true 200-step source (`0.2800`/`0.2750`). Geometry was mixed: forced-result ranking was strong (`forced_best_true=0.6725`) but 50-step slope final loss was `1.4660`, worse than scheduled forced-true (`1.0360`). The trusted 600-step frozen-policy additive handoff resolved the conflict positively: final eval `0.6600`, step-600 normal `0.7050`, injection-zero `0.0000`, forced-random `0.0250`, learned calc `0.3375`, beating the matched scheduled forced-true handoff (`0.4150`) and baseline (`0.2525`).
Do not repeat: Do not rerun the same seed-13, `operand_max=19`, 200-step one-negative forced-margin source plus 600-step handoff as novelty, and do not run the 4-negative full-grid branch without a compute-reduction plan.
Next allowed test: Extend the one-negative forced-margin source to longer horizons (`400/600`) and verify with trusted 600-step handoff, or replicate on a fresh seed if the explicit question is stability. Keep slope/geometry as diagnostics only; actual handoff remains arbiter.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-op19-gate.md`
```
