# The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-long-adaptation-selector-probe.md

Summary:

- Starting from the existing step-1500 800-step frozen handoff and adapting for 1600 steps with policy backbone frozen reached final eval `0.9100`, below the handoff-probe-selected step-1100 result `0.9250`, despite higher final calc accuracy (`0.9325` vs `0.8275`).

Questions:

- What did we learn about The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate?
- Has The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate been tested?
- Should we repeat The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate?
- What is the status of The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate?
- Why did The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-long-adaptation-selector-probe.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src5` step-1500 800-step frozen handoff into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.

Next Allowed:

- Compare against the exact old final-source checkpoint lineage, inspect reproduced-versus-old final-source differences, or optimize source acquisition for downstream readout compatibility.

Full Text:

```text
DISPROVEN: The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate.
Conclusion: Starting from the existing step-1500 800-step frozen handoff and adapting for 1600 steps with policy backbone frozen reached final eval `0.9100`, below the handoff-probe-selected step-1100 result `0.9250`, despite higher final calc accuracy (`0.9325` vs `0.8275`).
Do not repeat: Same `src5` step-1500 800-step frozen handoff into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Compare against the exact old final-source checkpoint lineage, inspect reproduced-versus-old final-source differences, or optimize source acquisition for downstream readout compatibility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-long-adaptation-selector-probe.md`
```
