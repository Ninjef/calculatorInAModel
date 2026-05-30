# Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-long-source-gate.md

Summary:

- Extending the one-negative forced-margin source branch on `operand_max=19`, seed-13 improved source accuracy and handoff, but exposed checkpoint/RNG sensitivity. A fresh 600-step source run reached `0.5225` train calc / `0.4800` final source eval with near-perfect geometry (`forced_best_true=0.9925`), and its step-600 checkpoint reached `0.7330` final eval / `0.7500` step-600 normal under trusted frozen-policy handoff with injection-zero `0.0000`, forced-random `0.0225`, learned calc `0.4975`. Continuing the exact prior positive step-200 checkpoint gave a better intermediate source checkpoint after 200 continuation steps (`0.4725` calc, `forced_best_true=0.9675`) whose handoff reached `0.7400` final eval / `0.7850` step-600 normal with injection-zero `0.0000`, forced-random `0.0300`, learned calc `0.4175`; continuing to 400 steps degraded source final eval back to `0.3600`. This improves over the one-negative 200-step handoff (`0.6600`) but does not clearly beat scheduled forced-true step-600 final handoff (`0.7725`).

Questions:

- What did we learn about Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600?
- Has Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600 been tested?
- Should we repeat Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600?
- What is the status of Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600?
- Why did Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600 fail?
- What follow-up is allowed for Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-long-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same seed-13 one-negative forced-margin 600-step source ladder, the same continuation from step-200, or handoffs from the tested step-400/step-600/continued-step-200 checkpoints as novelty.

Next Allowed:

- Try a late source-recovery/retention phase for one-negative margin only if explicitly testing the source-policy bottleneck; otherwise compare on a fresh seed or move to less prescriptive/scalable assignment. Keep actual 600-step handoff as arbiter because slope/geometry stayed imperfect selectors.

Full Text:

```text
MIXED-POSITIVE: Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600.
Conclusion: Extending the one-negative forced-margin source branch on `operand_max=19`, seed-13 improved source accuracy and handoff, but exposed checkpoint/RNG sensitivity. A fresh 600-step source run reached `0.5225` train calc / `0.4800` final source eval with near-perfect geometry (`forced_best_true=0.9925`), and its step-600 checkpoint reached `0.7330` final eval / `0.7500` step-600 normal under trusted frozen-policy handoff with injection-zero `0.0000`, forced-random `0.0225`, learned calc `0.4975`. Continuing the exact prior positive step-200 checkpoint gave a better intermediate source checkpoint after 200 continuation steps (`0.4725` calc, `forced_best_true=0.9675`) whose handoff reached `0.7400` final eval / `0.7850` step-600 normal with injection-zero `0.0000`, forced-random `0.0300`, learned calc `0.4175`; continuing to 400 steps degraded source final eval back to `0.3600`. This improves over the one-negative 200-step handoff (`0.6600`) but does not clearly beat scheduled forced-true step-600 final handoff (`0.7725`).
Do not repeat: Do not rerun the same seed-13 one-negative forced-margin 600-step source ladder, the same continuation from step-200, or handoffs from the tested step-400/step-600/continued-step-200 checkpoints as novelty.
Next allowed test: Try a late source-recovery/retention phase for one-negative margin only if explicitly testing the source-policy bottleneck; otherwise compare on a fresh seed or move to less prescriptive/scalable assignment. Keep actual 600-step handoff as arbiter because slope/geometry stayed imperfect selectors.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-long-source-gate.md`
```
