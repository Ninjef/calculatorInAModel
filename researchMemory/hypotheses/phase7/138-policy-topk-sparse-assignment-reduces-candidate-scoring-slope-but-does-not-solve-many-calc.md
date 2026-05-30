# Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-many-calculator-scaling-accounting.md

Summary:

- Added `scripts/analyze_assignment_scaling.py` to make the scorer and result-head accounting reproducible. For op29 over 630 assignment steps and the full `900`-prompt grid, exact hard assignment costs `33,453,000` forced result evaluations per calculator, while topk8+unique24 costs `13,608,000`; at 16 independent calculators this is `535,248,000` versus `217,728,000`. At op39, the same accounting is `79,632,000` exact versus `24,192,000` sampled per calculator, and `1,274,112,000` versus `387,072,000` at 16 calculators. Result-head parameters are linear too if calculators have independent `rhead64` heads (`12,091` each at op29, `13,391` each at op39). The current repo still implements one calculator hook, so this is an accounting/review result: topk changes the per-calculator result-class slope, but true many-calculator scalability still needs routing/multi-hook validation or a non-enumerative credit signal.

Questions:

- What did we learn about Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling?
- Has Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling been tested?
- Should we repeat Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling?
- What is the status of Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling?

Representative evidence:

- `researchReviews/2026-05-30-many-calculator-scaling-accounting.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat further op19/op29 policy-topk seed replications as many-calculator evidence, and do not claim topk solves scaling without an actual multi-calculator/routing implementation or active-calculator accounting.

Next Allowed:

- Implement a true multi-calculator/routed bottleneck diagnostic, stress op39 with a declared compute hypothesis, or replace hard assignment with a less-prescriptive/non-enumerative target construction.

Full Text:

```text
REVIEW: Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling.
Conclusion: Added `scripts/analyze_assignment_scaling.py` to make the scorer and result-head accounting reproducible. For op29 over 630 assignment steps and the full `900`-prompt grid, exact hard assignment costs `33,453,000` forced result evaluations per calculator, while topk8+unique24 costs `13,608,000`; at 16 independent calculators this is `535,248,000` versus `217,728,000`. At op39, the same accounting is `79,632,000` exact versus `24,192,000` sampled per calculator, and `1,274,112,000` versus `387,072,000` at 16 calculators. Result-head parameters are linear too if calculators have independent `rhead64` heads (`12,091` each at op29, `13,391` each at op39). The current repo still implements one calculator hook, so this is an accounting/review result: topk changes the per-calculator result-class slope, but true many-calculator scalability still needs routing/multi-hook validation or a non-enumerative credit signal.
Do not repeat: Do not treat further op19/op29 policy-topk seed replications as many-calculator evidence, and do not claim topk solves scaling without an actual multi-calculator/routing implementation or active-calculator accounting.
Next allowed test: Implement a true multi-calculator/routed bottleneck diagnostic, stress op39 with a declared compute hypothesis, or replace hard assignment with a less-prescriptive/non-enumerative target construction.
Source: `researchReviews/2026-05-30-many-calculator-scaling-accounting.md`
```
