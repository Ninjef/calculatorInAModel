# Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md

Summary:

- On seed 17, `additive_forced_true_loss <= 0.05` with min step `500` triggered at step `500`, reduced the late forced-true weight to `0.1`, and improved source final eval to `0.7225` versus `0.6100` for the no-trigger source. The trusted 600-step frozen-policy handoff reached `0.7625` final eval / `0.7825` step-600 snapshot with learned calc `0.7350`, injection-zero `0.0450`, and forced-random `0.0325`, close to the fixed step-600 control (`0.7675`) and above the raw source-accuracy trigger (`0.6825`), but still below the high gate.

Questions:

- What did we learn about Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate?
- Has Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate been tested?
- Should we repeat Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate?
- What is the status of Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-17 `additive_forced_true_loss <= 0.05`, min-step-500 adaptive recovery plus 600-step handoff as novelty.

Next Allowed:

- Use a smoothed/conjunctive transition criterion or move back toward scalable assignment; one raw metric can recover fixed-step timing but has not produced robust high-gate clears.

Full Text:

```text
MIXED: Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate.
Conclusion: On seed 17, `additive_forced_true_loss <= 0.05` with min step `500` triggered at step `500`, reduced the late forced-true weight to `0.1`, and improved source final eval to `0.7225` versus `0.6100` for the no-trigger source. The trusted 600-step frozen-policy handoff reached `0.7625` final eval / `0.7825` step-600 snapshot with learned calc `0.7350`, injection-zero `0.0450`, and forced-random `0.0325`, close to the fixed step-600 control (`0.7675`) and above the raw source-accuracy trigger (`0.6825`), but still below the high gate.
Do not repeat: The same seed-17 `additive_forced_true_loss <= 0.05`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Use a smoothed/conjunctive transition criterion or move back toward scalable assignment; one raw metric can recover fixed-step timing but has not produced robust high-gate clears.
Source: `aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md`
```
