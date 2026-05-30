# Static full-enum regret-set result-boundary targets improve source acquisition.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-regret-set-training-gate.md

Summary:

- Added `regret_set` target mode, a uniform target over forced result classes within a fixed NLL margin of the best forced result. Margins `0.05`, `0.25`, and `1.0` collapsed to hard-best (`1.0` effective results); margin `2.0` was still nearly hard (`1.06` effective results); margin `4.0` was genuinely set-valued (`5.6975` effective results, true result always in set, true-result target mass `0.2413`). But the margin-4 200-step source gate learned much worse than the matched hard-best comparator: regret-set step-200 learned calc / final eval `0.0900` / `0.0900` versus hard-best `0.4625` / `0.4225`. Simple static set-valued targets dilute the useful answer-derived signal instead of improving less-prescriptive source learning.

Questions:

- What did we learn about Static full-enum regret-set result-boundary targets improve source acquisition?
- Has Static full-enum regret-set result-boundary targets improve source acquisition been tested?
- Should we repeat Static full-enum regret-set result-boundary targets improve source acquisition?
- What is the status of Static full-enum regret-set result-boundary targets improve source acquisition?
- Why did Static full-enum regret-set result-boundary targets improve source acquisition fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-regret-set-training-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run fixed full-enum `regret_set` margin ladders or simple top-N-low-regret static target variants on this same gate as novelty.

Next Allowed:

- If staying with result-boundary, change the mechanism to adaptive/evolving validation or calibrated proposal learning; otherwise move to a different less-prescriptive credit-assignment family.

Full Text:

```text
DISPROVEN: Static full-enum regret-set result-boundary targets improve source acquisition.
Conclusion: Added `regret_set` target mode, a uniform target over forced result classes within a fixed NLL margin of the best forced result. Margins `0.05`, `0.25`, and `1.0` collapsed to hard-best (`1.0` effective results); margin `2.0` was still nearly hard (`1.06` effective results); margin `4.0` was genuinely set-valued (`5.6975` effective results, true result always in set, true-result target mass `0.2413`). But the margin-4 200-step source gate learned much worse than the matched hard-best comparator: regret-set step-200 learned calc / final eval `0.0900` / `0.0900` versus hard-best `0.4625` / `0.4225`. Simple static set-valued targets dilute the useful answer-derived signal instead of improving less-prescriptive source learning.
Do not repeat: Do not run fixed full-enum `regret_set` margin ladders or simple top-N-low-regret static target variants on this same gate as novelty.
Next allowed test: If staying with result-boundary, change the mechanism to adaptive/evolving validation or calibrated proposal learning; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-regret-set-training-gate.md`
```
