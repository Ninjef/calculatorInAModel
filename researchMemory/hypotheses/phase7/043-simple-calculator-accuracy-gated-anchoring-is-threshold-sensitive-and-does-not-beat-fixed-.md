# Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-accuracy-gated-anchor.md

Summary:

- Base anchor `0.01` with `current_argmax_accuracy` gates `<0.80` or `<0.82 -> 0.1` reached `src5` final eval `0.9825` at both thresholds, but `src4` reached only `0.7725/0.7900`, below fixed anchor `0.1` (`0.8325`).

Questions:

- What did we learn about Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`?
- Has Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1` been tested?
- Should we repeat Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`?
- What is the status of Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-accuracy-gated-anchor.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate thresholds `0.80` or `0.82`, gate weight `0.1`, calculator-accuracy gate, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Continuous/adaptive anchor control, selective policy-path unfreezing, stronger source acquisition, or a retention signal that combines calculator accuracy with answer utility.

Full Text:

```text
MIXED: Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with `current_argmax_accuracy` gates `<0.80` or `<0.82 -> 0.1` reached `src5` final eval `0.9825` at both thresholds, but `src4` reached only `0.7725/0.7900`, below fixed anchor `0.1` (`0.8325`).
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate thresholds `0.80` or `0.82`, gate weight `0.1`, calculator-accuracy gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Continuous/adaptive anchor control, selective policy-path unfreezing, stronger source acquisition, or a retention signal that combines calculator accuracy with answer utility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-accuracy-gated-anchor.md`
```
