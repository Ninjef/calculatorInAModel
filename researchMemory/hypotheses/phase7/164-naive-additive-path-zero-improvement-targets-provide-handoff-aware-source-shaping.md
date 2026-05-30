# Naive additive-path zero-improvement targets provide handoff-aware source shaping.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-additive-zero-improvement-source-gate.md

Summary:

- Added `result_boundary_target_mode=additive_zero_improvement`, which builds the zero-improvement target from forced-result answer-loss gains through the non-bottleneck additive path. In the 200-step full-enum source gate it learned the additive-path target (`learned_best=0.6025`) but the target itself was non-arithmetic (`hard_best_equals_true_sum=0.0325`, true-result target probability `0.0225`) and calculator-result accuracy stayed near chance (`0.0200` final / snapshot). The untrained additive/readout path creates arbitrary answer-derived result preferences, so this is not a viable handoff-aware shaping signal by itself.

Questions:

- What did we learn about Naive additive-path zero-improvement targets provide handoff-aware source shaping?
- Has Naive additive-path zero-improvement targets provide handoff-aware source shaping been tested?
- Should we repeat Naive additive-path zero-improvement targets provide handoff-aware source shaping?
- What is the status of Naive additive-path zero-improvement targets provide handoff-aware source shaping?
- Why did Naive additive-path zero-improvement targets provide handoff-aware source shaping fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-additive-zero-improvement-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run longer source/handoff jobs with plain `additive_zero_improvement` from an untrained additive readout as novelty; it first needs a mechanism that makes the additive forced-result loss table meaningful without true-result forcing.

Next Allowed:

- If using additive-path targets, add a real readout-preconditioning or co-training mechanism and predeclare how it avoids simply reintroducing prescriptive true-result supervision.

Full Text:

```text
DISPROVEN: Naive additive-path zero-improvement targets provide handoff-aware source shaping.
Conclusion: Added `result_boundary_target_mode=additive_zero_improvement`, which builds the zero-improvement target from forced-result answer-loss gains through the non-bottleneck additive path. In the 200-step full-enum source gate it learned the additive-path target (`learned_best=0.6025`) but the target itself was non-arithmetic (`hard_best_equals_true_sum=0.0325`, true-result target probability `0.0225`) and calculator-result accuracy stayed near chance (`0.0200` final / snapshot). The untrained additive/readout path creates arbitrary answer-derived result preferences, so this is not a viable handoff-aware shaping signal by itself.
Do not repeat: Do not run longer source/handoff jobs with plain `additive_zero_improvement` from an untrained additive readout as novelty; it first needs a mechanism that makes the additive forced-result loss table meaningful without true-result forcing.
Next allowed test: If using additive-path targets, add a real readout-preconditioning or co-training mechanism and predeclare how it avoids simply reintroducing prescriptive true-result supervision.
Source: `aiAgentWorkHistory/phase7/2026-05-30-additive-zero-improvement-source-gate.md`
```
