# EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md

Summary:

- Adding trigger EMA/patience support and running `additive_forced_true_loss <= 0.05` with EMA beta `0.8`, patience `10`, and min step `500` fired at step `509`. Source final eval reached `0.7625`, and trusted 600-step frozen-policy handoff reached `0.8025` final eval / `0.7975` step-600 snapshot with learned calc `0.7425`, injection-zero `0.0625`, and forced-random `0.0325`. This beats raw forced-loss trigger (`0.7625` handoff), fixed step-600 (`0.7675`), and raw source-accuracy trigger (`0.6825`), but remains below the high gate.

Questions:

- What did we learn about EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate?
- Has EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate been tested?
- Should we repeat EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate?
- What is the status of EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate?
- What follow-up is allowed for EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-17 forced-loss `threshold=0.05`, EMA beta `0.8`, patience `10`, min-step-500 adaptive recovery plus 600-step handoff as novelty.

Next Allowed:

- Try a conjunctive source-plus-geometry trigger or return to scalable assignment; smoothing/patience helps timing but is not sufficient by itself.

Full Text:

```text
MIXED-POSITIVE: EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate.
Conclusion: Adding trigger EMA/patience support and running `additive_forced_true_loss <= 0.05` with EMA beta `0.8`, patience `10`, and min step `500` fired at step `509`. Source final eval reached `0.7625`, and trusted 600-step frozen-policy handoff reached `0.8025` final eval / `0.7975` step-600 snapshot with learned calc `0.7425`, injection-zero `0.0625`, and forced-random `0.0325`. This beats raw forced-loss trigger (`0.7625` handoff), fixed step-600 (`0.7675`), and raw source-accuracy trigger (`0.6825`), but remains below the high gate.
Do not repeat: The same seed-17 forced-loss `threshold=0.05`, EMA beta `0.8`, patience `10`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Try a conjunctive source-plus-geometry trigger or return to scalable assignment; smoothing/patience helps timing but is not sufficient by itself.
Source: `aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md`
```
