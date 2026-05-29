# The seed-10 transfer failure is only a bad final-checkpoint selection artifact.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md

Summary:

- Earlier seed-10 checkpoints improved 600-step handoff over final (`0.4475/0.4325/0.4225` vs `0.3375`), but all stayed below seed-9 final reference (`0.5250` at 600 and `0.6500` final eval); frozen-state linear probing was not a valid selector because it ranked seed-10 final highest (`0.4500`) despite worst handoff.

Questions:

- What did we learn about The seed-10 transfer failure is only a bad final-checkpoint selection artifact?
- Has The seed-10 transfer failure is only a bad final-checkpoint selection artifact been tested?
- Should we repeat The seed-10 transfer failure is only a bad final-checkpoint selection artifact?
- What is the status of The seed-10 transfer failure is only a bad final-checkpoint selection artifact?
- Why did The seed-10 transfer failure is only a bad final-checkpoint selection artifact fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same seed-10 step `1000`/`1300`/`1400` 600-step handoff sweep or frozen-state linear probe over these checkpoints as novelty.

Next Allowed:

- Build an additive learning-slope or injection-to-answer geometry proxy, or optimize source acquisition for early handoff slope.

Full Text:

```text
MIXED-NEGATIVE: The seed-10 transfer failure is only a bad final-checkpoint selection artifact.
Conclusion: Earlier seed-10 checkpoints improved 600-step handoff over final (`0.4475/0.4325/0.4225` vs `0.3375`), but all stayed below seed-9 final reference (`0.5250` at 600 and `0.6500` final eval); frozen-state linear probing was not a valid selector because it ranked seed-10 final highest (`0.4500`) despite worst handoff.
Do not repeat: Same seed-10 step `1000`/`1300`/`1400` 600-step handoff sweep or frozen-state linear probe over these checkpoints as novelty.
Next allowed test: Build an additive learning-slope or injection-to-answer geometry proxy, or optimize source acquisition for early handoff slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`
```
