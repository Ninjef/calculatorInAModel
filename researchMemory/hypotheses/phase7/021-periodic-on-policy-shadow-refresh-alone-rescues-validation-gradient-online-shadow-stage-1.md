# Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-on-policy-refresh-gate.md

Summary:

- Refresh every `50` steps restored excellent current-model heldout gradient agreement (`0.982-0.998` result cosine, ~`1.0` upstream), but Stage 1 ended at `0.025` final exact match with best snapshot `0.0475`.

Questions:

- What did we learn about Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1?
- Has Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1 been tested?
- Should we repeat Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1?
- What is the status of Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1?
- Why did Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1 fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-on-policy-refresh-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same h32 validation-gradient module with refresh every `50`, `shadow_feedback_weight=1.0`, no apply clamp, and 200-step budget as novelty.

Next Allowed:

- Add training-dynamics constraints such as step-level trust region, entropy/diversity stabilization, or a target/state that avoids single-result collapse.

Full Text:

```text
DISPROVEN: Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1.
Conclusion: Refresh every `50` steps restored excellent current-model heldout gradient agreement (`0.982-0.998` result cosine, ~`1.0` upstream), but Stage 1 ended at `0.025` final exact match with best snapshot `0.0475`.
Do not repeat: Same h32 validation-gradient module with refresh every `50`, `shadow_feedback_weight=1.0`, no apply clamp, and 200-step budget as novelty.
Next allowed test: Add training-dynamics constraints such as step-level trust region, entropy/diversity stabilization, or a target/state that avoids single-result collapse.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-on-policy-refresh-gate.md`
```
