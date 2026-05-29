# A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md

Summary:

- Adding an optional secondary adaptive trigger and requiring forced-loss readiness (`additive_forced_true_loss <= 0.05`, EMA beta `0.8`, patience `10`, min step `500`) plus `result_policy_argmax_result_accuracy >= 0.70` never activated recovery on seed 17. The primary forced-loss condition was ready for `132` consecutive logged steps and ended with EMA `0.0055`, but the secondary source-accuracy metric ended at `0.6325`; source final stayed `0.6100`, and trusted 600-step frozen-policy handoff reached only `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`.

Questions:

- What did we learn about A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger?
- Has A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger been tested?
- Should we repeat A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger?
- What is the status of A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger?
- Why did A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-17 forced-loss EMA/patience plus secondary source-accuracy `>=0.70` conjunctive recovery trigger and 600-step handoff as novelty.

Next Allowed:

- Do not keep tuning hard source-accuracy gates. Return to scalable assignment or source objectives that improve handoff/readout geometry directly; if another adaptive transition is tried, it needs a new signal family or a predeclared reason it should avoid the seed-17 no-fire failure.

Full Text:

```text
MIXED-NEGATIVE: A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger.
Conclusion: Adding an optional secondary adaptive trigger and requiring forced-loss readiness (`additive_forced_true_loss <= 0.05`, EMA beta `0.8`, patience `10`, min step `500`) plus `result_policy_argmax_result_accuracy >= 0.70` never activated recovery on seed 17. The primary forced-loss condition was ready for `132` consecutive logged steps and ended with EMA `0.0055`, but the secondary source-accuracy metric ended at `0.6325`; source final stayed `0.6100`, and trusted 600-step frozen-policy handoff reached only `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`.
Do not repeat: The same seed-17 forced-loss EMA/patience plus secondary source-accuracy `>=0.70` conjunctive recovery trigger and 600-step handoff as novelty.
Next allowed test: Do not keep tuning hard source-accuracy gates. Return to scalable assignment or source objectives that improve handoff/readout geometry directly; if another adaptive transition is tried, it needs a new signal family or a predeclared reason it should avoid the seed-17 no-fire failure.
Source: `aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md`
```
