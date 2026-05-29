# Appending raw result policy-state features rescues target-normalized online MLP shadow feedback.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-policy-state-gate.md

Summary:

- Adding result probabilities, log-probabilities, and entropy to the shadow input did not clear the heldout gap gate; `h32` reached heldout `0.7037/0.7611` but gaps were `0.2853/0.2131`, and `h16` missed the result threshold.

Questions:

- What did we learn about Appending raw result policy-state features rescues target-normalized online MLP shadow feedback?
- Has Appending raw result policy-state features rescues target-normalized online MLP shadow feedback been tested?
- Should we repeat Appending raw result policy-state features rescues target-normalized online MLP shadow feedback?
- What is the status of Appending raw result policy-state features rescues target-normalized online MLP shadow feedback?
- Why did Appending raw result policy-state features rescues target-normalized online MLP shadow feedback fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-policy-state-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Raw `injection_grad_policy_state` features with per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.

Next Allowed:

- Feature scaling/standardization, explicit regularization, a different synthetic-gradient loss, or a more stable target construction.

Full Text:

```text
DISPROVEN: Appending raw result policy-state features rescues target-normalized online MLP shadow feedback.
Conclusion: Adding result probabilities, log-probabilities, and entropy to the shadow input did not clear the heldout gap gate; `h32` reached heldout `0.7037/0.7611` but gaps were `0.2853/0.2131`, and `h16` missed the result threshold.
Do not repeat: Raw `injection_grad_policy_state` features with per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Feature scaling/standardization, explicit regularization, a different synthetic-gradient loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-policy-state-gate.md`
```
