# Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-continuous-anchor-gate.md

Summary:

- Linear gate with base `0.01`, threshold `0.85`, band `0.10`, and max `0.1` reached `src4` final eval `0.8375` with mean weight `0.0385`, but `src5` ended `0.9725`, slightly below fixed `0.1`/discrete accuracy gates.

Questions:

- What did we learn about Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement?
- Has Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement been tested?
- Should we repeat Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement?
- What is the status of Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-continuous-anchor-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, base `0.01`, linear `current_argmax_accuracy` gate threshold `0.85`, band `0.10`, gate weight `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Source-policy acquisition, selective policy-path unfreezing, or a retention controller that combines calculator accuracy with answer utility instead of metric-only anchor scaling.

Full Text:

```text
PARTIAL: Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement.
Conclusion: Linear gate with base `0.01`, threshold `0.85`, band `0.10`, and max `0.1` reached `src4` final eval `0.8375` with mean weight `0.0385`, but `src5` ended `0.9725`, slightly below fixed `0.1`/discrete accuracy gates.
Do not repeat: Same adapted `src4_add2/src5_add5`, base `0.01`, linear `current_argmax_accuracy` gate threshold `0.85`, band `0.10`, gate weight `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Source-policy acquisition, selective policy-path unfreezing, or a retention controller that combines calculator accuracy with answer utility instead of metric-only anchor scaling.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-continuous-anchor-gate.md`
```
