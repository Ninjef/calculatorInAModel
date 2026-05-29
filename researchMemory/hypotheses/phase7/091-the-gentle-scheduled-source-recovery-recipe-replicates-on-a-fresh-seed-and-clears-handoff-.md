# The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md

Summary:

- Fresh seed-14 scheduled source training reached step-600 source eval `0.6675`; the same 30-step CPU recovery (`lr=3e-4`, forced-true weight `0.1`) raised source eval to `0.8850`, and the trusted 600-step frozen-policy additive handoff reached `0.9600` final eval / `0.9650` step-600 snapshot with learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`.

Questions:

- What did we learn about The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly?
- Has The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly been tested?
- Should we repeat The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly?
- What is the status of The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly?
- What follow-up is allowed for The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-14 scheduled source -> 30-step low-LR recovery -> 600-step frozen-policy handoff as novelty.

Next Allowed:

- Automate the late-source transition or test a third seed only if the explicit question is stability; keep the 600-step handoff/readout gates as arbiter and monitor the somewhat higher seed-14 zero/random controls.

Full Text:

```text
POSITIVE: The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly.
Conclusion: Fresh seed-14 scheduled source training reached step-600 source eval `0.6675`; the same 30-step CPU recovery (`lr=3e-4`, forced-true weight `0.1`) raised source eval to `0.8850`, and the trusted 600-step frozen-policy additive handoff reached `0.9600` final eval / `0.9650` step-600 snapshot with learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`.
Do not repeat: The same seed-14 scheduled source -> 30-step low-LR recovery -> 600-step frozen-policy handoff as novelty.
Next allowed test: Automate the late-source transition or test a third seed only if the explicit question is stability; keep the 600-step handoff/readout gates as arbiter and monitor the somewhat higher seed-14 zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`
```
