# Static soft result-boundary targets improve source acquisition over hard-best targets.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-soft-target-training-gate.md

Summary:

- Tested existing `soft_result` result-boundary target construction on the matched full-grid upstream-open 200-step source gate. Temperature probe showed `t=1` is meaningfully soft (`0.8003` true-result target mass, `2.72` effective results), while `t=4` is broad (`0.1336` true mass, `28.35` effective results). Training was worse than the matched hard-best comparator: hard-best step-200 learned calc `0.5450` / final eval `0.5475`; soft `t=1` learned calc `0.2900` / final eval `0.2775`; soft `t=4` learned calc `0.1350` / final eval `0.1275`. Simple temperature-softened full-enum targets tolerate uncertainty by diluting the signal, not by improving scalable source learning.

Questions:

- What did we learn about Static soft result-boundary targets improve source acquisition over hard-best targets?
- Has Static soft result-boundary targets improve source acquisition over hard-best targets been tested?
- Should we repeat Static soft result-boundary targets improve source acquisition over hard-best targets?
- What is the status of Static soft result-boundary targets improve source acquisition over hard-best targets?
- Why did Static soft result-boundary targets improve source acquisition over hard-best targets fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-soft-target-training-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run static `soft_result` temperature ladders on the same full-grid result-boundary source gate as novelty.

Next Allowed:

- If using result-boundary targets, change the mechanism materially: uncertainty/regret-based set targets, evolving-checkpoint proposal validation, or a proposal model that reduces enumeration without merely softening the full-enum teacher.

Full Text:

```text
DISPROVEN: Static soft result-boundary targets improve source acquisition over hard-best targets.
Conclusion: Tested existing `soft_result` result-boundary target construction on the matched full-grid upstream-open 200-step source gate. Temperature probe showed `t=1` is meaningfully soft (`0.8003` true-result target mass, `2.72` effective results), while `t=4` is broad (`0.1336` true mass, `28.35` effective results). Training was worse than the matched hard-best comparator: hard-best step-200 learned calc `0.5450` / final eval `0.5475`; soft `t=1` learned calc `0.2900` / final eval `0.2775`; soft `t=4` learned calc `0.1350` / final eval `0.1275`. Simple temperature-softened full-enum targets tolerate uncertainty by diluting the signal, not by improving scalable source learning.
Do not repeat: Do not run static `soft_result` temperature ladders on the same full-grid result-boundary source gate as novelty.
Next allowed test: If using result-boundary targets, change the mechanism materially: uncertainty/regret-based set targets, evolving-checkpoint proposal validation, or a proposal model that reduces enumeration without merely softening the full-enum teacher.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-soft-target-training-gate.md`
```
