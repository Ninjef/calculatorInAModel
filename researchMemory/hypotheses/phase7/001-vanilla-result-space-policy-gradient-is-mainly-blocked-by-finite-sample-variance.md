# Vanilla result-space policy gradient is mainly blocked by finite-sample variance.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md

Summary:

- Exact result-marginal gradients align with sampled PG but both anti-align with the boundary ceiling.

Questions:

- What did we learn about Vanilla result-space policy gradient is mainly blocked by finite-sample variance?
- Has Vanilla result-space policy gradient is mainly blocked by finite-sample variance been tested?
- Should we repeat Vanilla result-space policy gradient is mainly blocked by finite-sample variance?
- What is the status of Vanilla result-space policy gradient is mainly blocked by finite-sample variance?
- Why did Vanilla result-space policy gradient is mainly blocked by finite-sample variance fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Longer vanilla PG or learned-baseline runs that estimate the same raw expected-cost gradient.

Next Allowed:

- A qualitatively different backward channel with a fixed-grid alignment gate.

Full Text:

```text
DISPROVEN: Vanilla result-space policy gradient is mainly blocked by finite-sample variance.
Conclusion: Exact result-marginal gradients align with sampled PG but both anti-align with the boundary ceiling.
Do not repeat: Longer vanilla PG or learned-baseline runs that estimate the same raw expected-cost gradient.
Next allowed test: A qualitatively different backward channel with a fixed-grid alignment gate.
Source: `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`
```
