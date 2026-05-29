# Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-result-policy-soft-diversity-gate.md

Summary:

- Low diversity weight `1.0` still collapsed to one hard result with final exact `0.015` unbounded and `0.005` clamped; high diversity weight `100` plus clamp `10` kept hard usage broader (`9.14` effective hard results) but reached only `0.070` final and `0.080` best snapshot.

Questions:

- What did we learn about Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1?
- Has Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1 been tested?
- Should we repeat Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1?
- What is the status of Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1?
- Why did Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1 fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-result-policy-soft-diversity-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same refreshed h32 validation-gradient module with soft result-policy diversity weights `1` or `100`, optional tiny entropy, 200-step budget, and clamp `0/10` as novelty.

Next Allowed:

- A hard/assignment-style usage constraint, step-level trust region, Jacobian-conditioned state, or richer target that links diverse requests to per-example improvement.

Full Text:

```text
DISPROVEN: Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1.
Conclusion: Low diversity weight `1.0` still collapsed to one hard result with final exact `0.015` unbounded and `0.005` clamped; high diversity weight `100` plus clamp `10` kept hard usage broader (`9.14` effective hard results) but reached only `0.070` final and `0.080` best snapshot.
Do not repeat: Same refreshed h32 validation-gradient module with soft result-policy diversity weights `1` or `100`, optional tiny entropy, 200-step budget, and clamp `0/10` as novelty.
Next allowed test: A hard/assignment-style usage constraint, step-level trust region, Jacobian-conditioned state, or richer target that links diverse requests to per-example improvement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-result-policy-soft-diversity-gate.md`
```
