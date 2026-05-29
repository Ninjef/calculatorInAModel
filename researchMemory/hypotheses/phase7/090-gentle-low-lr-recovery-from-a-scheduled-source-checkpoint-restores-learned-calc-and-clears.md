# Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md

Summary:

- Continuing the scheduled step-600 source for 30 CPU steps with LR `3e-4` and lower forced-true weight `0.1` raised source calc from `0.5800` to `0.7950` while keeping forced-true loss low; the resulting frozen-policy 600-step handoff reached `0.8425`, 800-step continuation reached `0.8900` final / `0.9175` best snapshot, and 600-step readout reached `0.9320` final eval with final diagnostic controls normal `0.9225`, injection-zero `0.0300`, forced-random `0.0325`, learned calc `0.7925`.

Questions:

- What did we learn about Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout?
- Has Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout been tested?
- Should we repeat Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout?
- What is the status of Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout?
- What follow-up is allowed for Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-13 scheduled step-600 -> 30-step low-LR `aux=0.1` recovery -> 600 handoff -> 800 continuation -> 600 readout chain as novelty.

Next Allowed:

- Replicate on a fresh scheduled source seed or integrate the low-LR/lower-aux recovery as an automatic late-source phase, then verify with the trusted 600-step handoff and continuation/readout gates.

Full Text:

```text
POSITIVE: Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout.
Conclusion: Continuing the scheduled step-600 source for 30 CPU steps with LR `3e-4` and lower forced-true weight `0.1` raised source calc from `0.5800` to `0.7950` while keeping forced-true loss low; the resulting frozen-policy 600-step handoff reached `0.8425`, 800-step continuation reached `0.8900` final / `0.9175` best snapshot, and 600-step readout reached `0.9320` final eval with final diagnostic controls normal `0.9225`, injection-zero `0.0300`, forced-random `0.0325`, learned calc `0.7925`.
Do not repeat: The same seed-13 scheduled step-600 -> 30-step low-LR `aux=0.1` recovery -> 600 handoff -> 800 continuation -> 600 readout chain as novelty.
Next allowed test: Replicate on a fresh scheduled source seed or integrate the low-LR/lower-aux recovery as an automatic late-source phase, then verify with the trusted 600-step handoff and continuation/readout gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`
```
