# Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md

Summary:

- In a 200-step adaptive proposal gate, raw uniform `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` exact-grid calc and `0.3438` sampled normal. Adaptive low-loss-neighborhood branches underperformed at similar raw scoring budgets: `u8_b4_r2` `0.2025` calc, `u8_b4_r3` `0.2600`, and `u12_b4_r2` `0.2700`; the adaptive branches had lower unique coverage (`18.42-22.08` unique results) and lower true-result coverage (`0.6350-0.7700`) than raw `u32` (`32` unique, `0.8450` coverage).

Questions:

- What did we learn about Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets?
- Has Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets been tested?
- Should we repeat Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets?
- What is the status of Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets?
- Why did Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 200-step adaptive neighborhood gate over `u8_b4_r2/u8_b4_r3/u12_b4_r2` as novelty.

Next Allowed:

- Use a learned proposal or importance/bias-corrected sampled target; otherwise pivot to source-acquisition-for-handoff geometry instead of more local sampled-candidate variants.

Full Text:

```text
DISPROVEN: Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets.
Conclusion: In a 200-step adaptive proposal gate, raw uniform `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` exact-grid calc and `0.3438` sampled normal. Adaptive low-loss-neighborhood branches underperformed at similar raw scoring budgets: `u8_b4_r2` `0.2025` calc, `u8_b4_r3` `0.2600`, and `u12_b4_r2` `0.2700`; the adaptive branches had lower unique coverage (`18.42-22.08` unique results) and lower true-result coverage (`0.6350-0.7700`) than raw `u32` (`32` unique, `0.8450` coverage).
Do not repeat: The same seed-2 200-step adaptive neighborhood gate over `u8_b4_r2/u8_b4_r3/u12_b4_r2` as novelty.
Next allowed test: Use a learned proposal or importance/bias-corrected sampled target; otherwise pivot to source-acquisition-for-handoff geometry instead of more local sampled-candidate variants.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`
```
