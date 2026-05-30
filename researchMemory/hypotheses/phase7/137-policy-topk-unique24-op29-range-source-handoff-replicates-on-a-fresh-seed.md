# Policy-topk unique24 op29 range source/handoff replicates on a fresh seed.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-fresh-seed-validation.md

Summary:

- Repeated the `topk8+unique24` op29 `rhead64` staged recipe on CLI seed `31` / effective seed `33`, matching the exact full-grid fresh-range comparator seed while still scoring only `24/59` result classes per assignment. The source reached `899/900 = 0.9989` final eval and step-630 normal/source-calc `0.9989`, with injection-zero `0.0200`, oracle `1.0000`, and forced-random `0.0133`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal `1.0000` / learned-calc `0.9989`, with injection-zero `0.0333`, oracle `1.0000`, and forced-random `0.0111`; final 128-sample metrics reported learned calc `1.0000`. This upgrades policy-aware sparse assignment from one-seed op29 validation to replicated op29 range evidence, while preserving the caveat that the method still uses hard assignment, forced-margin source shaping, pretrained product decoder, hidden result-head capacity, and frozen transfer.

Questions:

- What did we learn about Policy-topk unique24 op29 range source/handoff replicates on a fresh seed?
- Has Policy-topk unique24 op29 range source/handoff replicates on a fresh seed been tested?
- Should we repeat Policy-topk unique24 op29 range source/handoff replicates on a fresh seed?
- What is the status of Policy-topk unique24 op29 range source/handoff replicates on a fresh seed?
- What follow-up is allowed for Policy-topk unique24 op29 range source/handoff replicates on a fresh seed?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-fresh-seed-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more op29 `rhead64` topk8+unique24 source630 plus trusted handoff seed replications as novelty; effective seeds `29` and `33` have now cleared this range axis.

Next Allowed:

- Move to many-calculator cost/accounting, op39 range with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.

Full Text:

```text
POSITIVE: Policy-topk unique24 op29 range source/handoff replicates on a fresh seed.
Conclusion: Repeated the `topk8+unique24` op29 `rhead64` staged recipe on CLI seed `31` / effective seed `33`, matching the exact full-grid fresh-range comparator seed while still scoring only `24/59` result classes per assignment. The source reached `899/900 = 0.9989` final eval and step-630 normal/source-calc `0.9989`, with injection-zero `0.0200`, oracle `1.0000`, and forced-random `0.0133`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal `1.0000` / learned-calc `0.9989`, with injection-zero `0.0333`, oracle `1.0000`, and forced-random `0.0111`; final 128-sample metrics reported learned calc `1.0000`. This upgrades policy-aware sparse assignment from one-seed op29 validation to replicated op29 range evidence, while preserving the caveat that the method still uses hard assignment, forced-margin source shaping, pretrained product decoder, hidden result-head capacity, and frozen transfer.
Do not repeat: Do not run more op29 `rhead64` topk8+unique24 source630 plus trusted handoff seed replications as novelty; effective seeds `29` and `33` have now cleared this range axis.
Next allowed test: Move to many-calculator cost/accounting, op39 range with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-fresh-seed-validation.md`
```
