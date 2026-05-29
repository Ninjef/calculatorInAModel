# Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-output-jacobian-shadow-feature-gate.md

Summary:

- `injection_grad_logits_output_jacobian` with feature z-scoring cleared Stage 0B at `0.9073/0.9011` heldout result/upstream cosines, but refreshed clamp-`10` Stage 1 ended at `0.055` final exact with best snapshot `0.065`.

Questions:

- What did we learn about Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1?
- Has Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1 been tested?
- Should we repeat Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1?
- What is the status of Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-output-jacobian-shadow-feature-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- h16/h32 raw output-Jacobian features or h32 fit-split feature z-scoring with validation-gradient `0.5`, norm `0.1`, refresh every `50`, clamp `10`, and 200-step budget as novelty.

Next Allowed:

- Hard assignment-style usage constraints, richer targets, or a more substantial learned-gradient update path; do not treat this state-only Jacobian feature as enough.

Full Text:

```text
PARTIAL: Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1.
Conclusion: `injection_grad_logits_output_jacobian` with feature z-scoring cleared Stage 0B at `0.9073/0.9011` heldout result/upstream cosines, but refreshed clamp-`10` Stage 1 ended at `0.055` final exact with best snapshot `0.065`.
Do not repeat: h16/h32 raw output-Jacobian features or h32 fit-split feature z-scoring with validation-gradient `0.5`, norm `0.1`, refresh every `50`, clamp `10`, and 200-step budget as novelty.
Next allowed test: Hard assignment-style usage constraints, richer targets, or a more substantial learned-gradient update path; do not treat this state-only Jacobian feature as enough.
Source: `aiAgentWorkHistory/phase7/2026-05-28-output-jacobian-shadow-feature-gate.md`
```
