# A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md

Summary:

- Stage 0 model-update alignment was very high (`0.9983` result-proj, `0.9854` upstream), but 200-step Stage 1 reached only `0.070` best snapshot accuracy and `0.040` final exact match.

Questions:

- What did we learn about A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery?
- Has A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery been tested?
- Should we repeat A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery?
- What is the status of A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery?
- Why did A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Frozen fit-once linear shadow feedback with the same exact-grid calibration and weight/schedule.

Next Allowed:

- Heldout-validated or online-trained shadow modules with an early-lift gate, not a fixed linear map fit once at initialization.

Full Text:

```text
DISPROVEN: A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery.
Conclusion: Stage 0 model-update alignment was very high (`0.9983` result-proj, `0.9854` upstream), but 200-step Stage 1 reached only `0.070` best snapshot accuracy and `0.040` final exact match.
Do not repeat: Frozen fit-once linear shadow feedback with the same exact-grid calibration and weight/schedule.
Next allowed test: Heldout-validated or online-trained shadow modules with an early-lift gate, not a fixed linear map fit once at initialization.
Source: `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`
```
