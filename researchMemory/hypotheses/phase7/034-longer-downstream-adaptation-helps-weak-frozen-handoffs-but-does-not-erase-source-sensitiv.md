# Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-downstream-adaptation.md

Summary:

- Continuing weak cells for another 800 steps improved `src5_add5` final eval `0.5550 -> 0.8175` and `src4_add2` `0.3025 -> 0.6050`, while injection-zero stayed near chance and learned calc stayed `0.8000/0.8725`.

Questions:

- What did we learn about Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity?
- Has Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity been tested?
- Should we repeat Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity?
- What is the status of Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-downstream-adaptation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same `src4_add2` or `src5_add5` one-extra-800-step continuation as novelty.

Next Allowed:

- Better source checkpoint selection, stronger readout adaptation, controlled unfreezing, or source-policy training that produces more handoff-friendly representations.

Full Text:

```text
PARTIAL: Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity.
Conclusion: Continuing weak cells for another 800 steps improved `src5_add5` final eval `0.5550 -> 0.8175` and `src4_add2` `0.3025 -> 0.6050`, while injection-zero stayed near chance and learned calc stayed `0.8000/0.8725`.
Do not repeat: The same `src4_add2` or `src5_add5` one-extra-800-step continuation as novelty.
Next allowed test: Better source checkpoint selection, stronger readout adaptation, controlled unfreezing, or source-policy training that produces more handoff-friendly representations.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-downstream-adaptation.md`
```
