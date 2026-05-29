# Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-retention-gate.md

Summary:

- Assignment weight `10 -> 0` over 200 steps with `answer_loss_weight=1` peaked at `0.370` before shutoff, was `0.3475` at step `200`, collapsed to `0.105` by step `250`, and ended at `0.1075`.

Questions:

- What did we learn about Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface?
- Has Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface been tested?
- Should we repeat Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface?
- What is the status of Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface?
- Why did Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-retention-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same seed-2/seed-4 exact-grid no-shadow assignment decay over `200` steps with 400-step budget as novelty.

Next Allowed:

- Longer always-on convergence, seed replication, a stronger handoff bridge, or lower-cost assignment approximation.

Full Text:

```text
DISPROVEN: Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface.
Conclusion: Assignment weight `10 -> 0` over 200 steps with `answer_loss_weight=1` peaked at `0.370` before shutoff, was `0.3475` at step `200`, collapsed to `0.105` by step `250`, and ended at `0.1075`.
Do not repeat: Same seed-2/seed-4 exact-grid no-shadow assignment decay over `200` steps with 400-step budget as novelty.
Next allowed test: Longer always-on convergence, seed replication, a stronger handoff bridge, or lower-cost assignment approximation.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-retention-gate.md`
```
