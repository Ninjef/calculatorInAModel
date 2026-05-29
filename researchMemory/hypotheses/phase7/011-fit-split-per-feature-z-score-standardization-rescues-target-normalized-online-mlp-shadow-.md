# Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-feature-standardization-gate.md

Summary:

- Feature z-scoring hurt the raw policy-state branch (`h16` heldout `0.5942/0.3997`, `h32` `0.4340/0.4023`) and did not rescue the simpler logits branch (`h32` heldout `0.6691/0.7028`, gaps `0.2830/0.2658`).

Questions:

- What did we learn about Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback?
- Has Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback been tested?
- Should we repeat Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback?
- What is the status of Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback?
- Why did Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-feature-standardization-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Plain `fit_zscore_per_feature` with `injection_grad_logits` or `injection_grad_policy_state`, per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.

Next Allowed:

- Change objective/regularization or target construction, not just raw feature scale.

Full Text:

```text
DISPROVEN: Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback.
Conclusion: Feature z-scoring hurt the raw policy-state branch (`h16` heldout `0.5942/0.3997`, `h32` `0.4340/0.4023`) and did not rescue the simpler logits branch (`h32` heldout `0.6691/0.7028`, gaps `0.2830/0.2658`).
Do not repeat: Plain `fit_zscore_per_feature` with `injection_grad_logits` or `injection_grad_policy_state`, per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Change objective/regularization or target construction, not just raw feature scale.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-feature-standardization-gate.md`
```
