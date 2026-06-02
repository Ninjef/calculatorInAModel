# Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-capped-prior-many-calculator-accounting.md

Summary:

- Added `scripts/analyze_prior_replay_scaling.py` and measured-family accounting for the op29 quality-gated capped-prior recipe. Per calculator, the original capped source uses about `294,912` sparse candidate evals, `1,254,817` prior fit examples, `1,080,000` full-fit examples, `720` prompt-memory entries, and only `7,995` numeric-prior parameters. At 16 independent op29 calculators this becomes `4,718,592` candidate evals plus `20,077,072` prior fit examples (`24,795,664` candidate+prior examples); at 64 calculators it becomes `99,182,656` candidate+prior examples. This is much cheaper than the old op29 topk8+unique24 hard-assignment accounting (`217,728,000` forced evals at 16 calculators), but it still fails the many-calculator scalability requirement because prompt memory and prior fitting are per-calculator and linear.

Questions:

- What did we learn about Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators?
- Has Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators been tested?
- Should we repeat Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators?
- What is the status of Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-capped-prior-many-calculator-accounting.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat cap/fraction/window/seed ladders as scalability progress; do not claim small prior parameter count solves training cost.

Next Allowed:

- Shared/global prior or target discovery across calculators, removal of per-calculator prompt-memory target tables, or a less-prescriptive credit mechanism that bypasses answer-derived candidate scoring.

Full Text:

```text
REVIEW: Capped numeric-prior replay improves candidate-scoring cost but still scales linearly with independent calculators.
Conclusion: Added `scripts/analyze_prior_replay_scaling.py` and measured-family accounting for the op29 quality-gated capped-prior recipe. Per calculator, the original capped source uses about `294,912` sparse candidate evals, `1,254,817` prior fit examples, `1,080,000` full-fit examples, `720` prompt-memory entries, and only `7,995` numeric-prior parameters. At 16 independent op29 calculators this becomes `4,718,592` candidate evals plus `20,077,072` prior fit examples (`24,795,664` candidate+prior examples); at 64 calculators it becomes `99,182,656` candidate+prior examples. This is much cheaper than the old op29 topk8+unique24 hard-assignment accounting (`217,728,000` forced evals at 16 calculators), but it still fails the many-calculator scalability requirement because prompt memory and prior fitting are per-calculator and linear.
Do not repeat: Do not treat cap/fraction/window/seed ladders as scalability progress; do not claim small prior parameter count solves training cost.
Next allowed test: Shared/global prior or target discovery across calculators, removal of per-calculator prompt-memory target tables, or a less-prescriptive credit mechanism that bypasses answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-02-capped-prior-many-calculator-accounting.md`
```
