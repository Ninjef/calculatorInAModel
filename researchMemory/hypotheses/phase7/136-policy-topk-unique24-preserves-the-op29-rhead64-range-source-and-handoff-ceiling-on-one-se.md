# Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-range-validation.md

Summary:

- Tested the replicated `topk8+unique24` sparse hard-assignment proposal on the op29 `rhead64` staged recipe, using the exact full-grid effective-seed-29 ceiling as the comparator. The sparse source scored only `24/59` result classes per assignment instead of exact `59/59`, yet reached `900/900 = 1.0000` final eval and step-630 normal/source-calc `1.0000`, with injection-zero `0.0233`, oracle `1.0000`, and forced-random `0.0144`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal/learned-calc `1.0000`, with injection-zero `0.0356`, oracle `1.0000`, and forced-random `0.0189`. This is the first operand-range validation that policy-aware sparse assignment can preserve an exact-grid source/handoff ceiling at much lower result-class scoring cost, but it remains one op29 seed and still uses hard assignment, forced-margin source shaping, a pretrained product decoder, hidden result-head capacity, and frozen transfer.

Questions:

- What did we learn about Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed?
- Has Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed been tested?
- Should we repeat Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed?
- What is the status of Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed?
- What follow-up is allowed for Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-range-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same effective-seed-29 op29 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty; it has already been compared to the exact ceiling on that seed.

Next Allowed:

- Validate this range result on a fresh op29 seed, stress op39/many-calculator cost with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.

Full Text:

```text
POSITIVE: Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed.
Conclusion: Tested the replicated `topk8+unique24` sparse hard-assignment proposal on the op29 `rhead64` staged recipe, using the exact full-grid effective-seed-29 ceiling as the comparator. The sparse source scored only `24/59` result classes per assignment instead of exact `59/59`, yet reached `900/900 = 1.0000` final eval and step-630 normal/source-calc `1.0000`, with injection-zero `0.0233`, oracle `1.0000`, and forced-random `0.0144`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal/learned-calc `1.0000`, with injection-zero `0.0356`, oracle `1.0000`, and forced-random `0.0189`. This is the first operand-range validation that policy-aware sparse assignment can preserve an exact-grid source/handoff ceiling at much lower result-class scoring cost, but it remains one op29 seed and still uses hard assignment, forced-margin source shaping, a pretrained product decoder, hidden result-head capacity, and frozen transfer.
Do not repeat: Do not rerun the same effective-seed-29 op29 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty; it has already been compared to the exact ceiling on that seed.
Next allowed test: Validate this range result on a fresh op29 seed, stress op39/many-calculator cost with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-range-validation.md`
```
