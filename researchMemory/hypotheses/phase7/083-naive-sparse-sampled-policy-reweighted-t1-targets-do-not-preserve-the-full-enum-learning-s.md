# Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md

Summary:

- In a 200-step no-replacement sparse approximation gate, exact `policy_reweighted_t1` reached `0.5600` exact-grid calc and `0.5391` sampled normal. Sparse uniform branches improved with coverage but lagged badly: `u16` reached `0.1975` calc, `u24` `0.2800`, `u32` `0.3350`, and near-full `u36` `0.4100`; only full-vocabulary `u39` recovered/exceeded the signal at `0.6250`. Top-k plus uniform was worse (`k8_u8` `0.0925` calc).

Questions:

- What did we learn about Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full?
- Has Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full been tested?
- Should we repeat Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full?
- What is the status of Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full?
- Why did Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 200-step sparse candidate ladder over `k8_u8/k0_u16/k0_u24/k0_u32/k0_u36/k0_u39` as novelty.

Next Allowed:

- Use a smarter proposal/learned candidate generator or importance-corrected target that improves true-result coverage without near-full forced-result enumeration.

Full Text:

```text
MIXED-NEGATIVE: Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full.
Conclusion: In a 200-step no-replacement sparse approximation gate, exact `policy_reweighted_t1` reached `0.5600` exact-grid calc and `0.5391` sampled normal. Sparse uniform branches improved with coverage but lagged badly: `u16` reached `0.1975` calc, `u24` `0.2800`, `u32` `0.3350`, and near-full `u36` `0.4100`; only full-vocabulary `u39` recovered/exceeded the signal at `0.6250`. Top-k plus uniform was worse (`k8_u8` `0.0925` calc).
Do not repeat: The same seed-2 200-step sparse candidate ladder over `k8_u8/k0_u16/k0_u24/k0_u32/k0_u36/k0_u39` as novelty.
Next allowed test: Use a smarter proposal/learned candidate generator or importance-corrected target that improves true-result coverage without near-full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`
```
