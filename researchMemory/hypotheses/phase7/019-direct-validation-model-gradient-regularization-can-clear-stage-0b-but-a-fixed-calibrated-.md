# Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md

Summary:

- h32/validation-gradient `0.5`/norm `0.1` reached heldout `0.8068/0.8083` with gaps `0.1227/0.1343` and norms `1.1276/1.0736`; fixed-module Stage 1 weights `1.0/0.01/0.001` ended at `0.075/0.005/0.035` exact match.

Questions:

- What did we learn about Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift?
- Has Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift been tested?
- Should we repeat Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift?
- What is the status of Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same h16/h32 validation-gradient `0.5`, norm `0/0.1` Stage 0B grid or fixed-module Stage 1 weights `1.0/0.01/0.001` as novelty.

Next Allowed:

- Keep the direct gradient objective, but refresh the shadow module on-policy, add trust-region/norm clamps, or condition on state that remains valid after model movement.

Full Text:

```text
PARTIAL: Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift.
Conclusion: h32/validation-gradient `0.5`/norm `0.1` reached heldout `0.8068/0.8083` with gaps `0.1227/0.1343` and norms `1.1276/1.0736`; fixed-module Stage 1 weights `1.0/0.01/0.001` ended at `0.075/0.005/0.035` exact match.
Do not repeat: The same h16/h32 validation-gradient `0.5`, norm `0/0.1` Stage 0B grid or fixed-module Stage 1 weights `1.0/0.01/0.001` as novelty.
Next allowed test: Keep the direct gradient objective, but refresh the shadow module on-policy, add trust-region/norm clamps, or condition on state that remains valid after model movement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`
```
