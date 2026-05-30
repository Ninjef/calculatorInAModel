# Preserving unscored policy mass does not rescue sparse policy-reweighted local targets.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-corrected-sparse-local-target-gate.md

Summary:

- Added `corrected_policy_reweighted_t<T>_u<U>_b<mean|current|max>`, which scores uniform candidates but imputes a baseline loss for unscored result classes instead of forcing their target mass to zero. In the 200-step full-grid gate, exact `policy_reweighted_t1` reached `0.5600` exact calc / `0.5391` sampled normal and raw `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` / `0.3438`; corrected branches were worse: `u8_bmean` `0.1150` / `0.0938`, `u8_bcurrent` `0.1100` / `0.0938`, `u8_bmax` `0.0675` / `0.0625`, `u16_bmean` `0.2100` / `0.2500`, and `u16_bcurrent` `0.2500` / `0.2500`. The correction diluted pressure and did not overcome low true-candidate coverage (`0.1850` for `u8`, `0.4050` for `u16`).

Questions:

- What did we learn about Preserving unscored policy mass does not rescue sparse policy-reweighted local targets?
- Has Preserving unscored policy mass does not rescue sparse policy-reweighted local targets been tested?
- Should we repeat Preserving unscored policy mass does not rescue sparse policy-reweighted local targets?
- What is the status of Preserving unscored policy mass does not rescue sparse policy-reweighted local targets?
- Why did Preserving unscored policy mass does not rescue sparse policy-reweighted local targets fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-corrected-sparse-local-target-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun corrected/imputed sparse targets with the same mean/current/max baselines or simply tune `u8/u16` sample counts as novelty.

Next Allowed:

- Local-target approximation still needs a learned/generalized proposal, a stronger estimator correction with an explicit bias/variance argument, or a target construction that creates useful pressure without requiring true-result coverage.

Full Text:

```text
DISPROVEN: Preserving unscored policy mass does not rescue sparse policy-reweighted local targets.
Conclusion: Added `corrected_policy_reweighted_t<T>_u<U>_b<mean|current|max>`, which scores uniform candidates but imputes a baseline loss for unscored result classes instead of forcing their target mass to zero. In the 200-step full-grid gate, exact `policy_reweighted_t1` reached `0.5600` exact calc / `0.5391` sampled normal and raw `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` / `0.3438`; corrected branches were worse: `u8_bmean` `0.1150` / `0.0938`, `u8_bcurrent` `0.1100` / `0.0938`, `u8_bmax` `0.0675` / `0.0625`, `u16_bmean` `0.2100` / `0.2500`, and `u16_bcurrent` `0.2500` / `0.2500`. The correction diluted pressure and did not overcome low true-candidate coverage (`0.1850` for `u8`, `0.4050` for `u16`).
Do not repeat: Do not rerun corrected/imputed sparse targets with the same mean/current/max baselines or simply tune `u8/u16` sample counts as novelty.
Next allowed test: Local-target approximation still needs a learned/generalized proposal, a stronger estimator correction with an explicit bias/variance argument, or a target construction that creates useful pressure without requiring true-result coverage.
Source: `aiAgentWorkHistory/phase7/2026-05-29-corrected-sparse-local-target-gate.md`
```
