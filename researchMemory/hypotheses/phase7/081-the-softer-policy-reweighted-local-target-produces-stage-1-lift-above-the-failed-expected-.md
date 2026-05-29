# The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md

Summary:

- In a 200-step exact-grid Stage 1 gate, `policy_reweighted_t1` reached `0.5600` exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy with controls low (`injection_zero=0.0234`, `forced_random=0.0156`), slightly above the hard-boundary ceiling run at the same budget (`0.5500` calc, `0.4844` normal). Ordinary expected loss collapsed to near chance (`0.0025` calc, `0.0000` normal), while `logit_descent_p0.1` improved but lagged (`0.2950` calc, `0.1953` normal).

Questions:

- What did we learn about The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline?
- Has The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline been tested?
- Should we repeat The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline?
- What is the status of The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline?
- What follow-up is allowed for The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2, 200-step Stage 1 comparison of `hard_boundary`, `expected_loss`, `policy_reweighted_t1`, and `logit_descent_p0.1` as novelty.

Next Allowed:

- Replicate or extend `policy_reweighted_t1` to a longer convergence/retention gate, then design a sampled/top-k/learned approximation that avoids full forced-result enumeration.

Full Text:

```text
PARTIAL-POSITIVE: The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline.
Conclusion: In a 200-step exact-grid Stage 1 gate, `policy_reweighted_t1` reached `0.5600` exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy with controls low (`injection_zero=0.0234`, `forced_random=0.0156`), slightly above the hard-boundary ceiling run at the same budget (`0.5500` calc, `0.4844` normal). Ordinary expected loss collapsed to near chance (`0.0025` calc, `0.0000` normal), while `logit_descent_p0.1` improved but lagged (`0.2950` calc, `0.1953` normal).
Do not repeat: The same seed-2, 200-step Stage 1 comparison of `hard_boundary`, `expected_loss`, `policy_reweighted_t1`, and `logit_descent_p0.1` as novelty.
Next allowed test: Replicate or extend `policy_reweighted_t1` to a longer convergence/retention gate, then design a sampled/top-k/learned approximation that avoids full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`
```
