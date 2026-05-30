# Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-adaptive-proposal-diagnostic.md

Summary:

- Added adaptive top-8-to-top-16 expansion metrics to the result-boundary critic diagnostic. On the trained step-800 result-boundary checkpoint, cutoff-margin expansion beat random expansion at matched average candidate count: single critic reached `0.85` vs random `0.82` at mean `10/39` candidates and `0.92` vs `0.88` at mean `12/39`; the four-critic ensemble reached `0.91` vs `0.88` at mean `10/39` and `0.97` vs `0.91` at mean `12/39`. However, fixed top-16 still reached `0.96` single-critic and `1.00` ensemble, the ensemble uses `32` train scores per prompt, and explicit std/LCB uncertainty scores were weaker than the simple margin heuristic. Adaptive expansion is better than random, but the current critic does not give enough adaptive-compute advantage to be the scalable result-boundary bridge.

Questions:

- What did we learn about Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage?
- Has Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage been tested?
- Should we repeat Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage?
- What is the status of Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage?
- Why did Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-adaptive-proposal-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not spend more mainline turns on threshold, beta, or expand-fraction sweeps over this same static fixed-grid diagnostic. The useful result is the margin-vs-random comparison and the remaining gap to fixed top-16.

Next Allowed:

- Change the mechanism: a calibrated proposal model validated across evolving checkpoints, a soft/set target that tolerates missing the exact argmin, or a source-training gate that uses a materially different target construction rather than static top-k expansion.

Full Text:

```text
MIXED-NEGATIVE: Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage.
Conclusion: Added adaptive top-8-to-top-16 expansion metrics to the result-boundary critic diagnostic. On the trained step-800 result-boundary checkpoint, cutoff-margin expansion beat random expansion at matched average candidate count: single critic reached `0.85` vs random `0.82` at mean `10/39` candidates and `0.92` vs `0.88` at mean `12/39`; the four-critic ensemble reached `0.91` vs `0.88` at mean `10/39` and `0.97` vs `0.91` at mean `12/39`. However, fixed top-16 still reached `0.96` single-critic and `1.00` ensemble, the ensemble uses `32` train scores per prompt, and explicit std/LCB uncertainty scores were weaker than the simple margin heuristic. Adaptive expansion is better than random, but the current critic does not give enough adaptive-compute advantage to be the scalable result-boundary bridge.
Do not repeat: Do not spend more mainline turns on threshold, beta, or expand-fraction sweeps over this same static fixed-grid diagnostic. The useful result is the margin-vs-random comparison and the remaining gap to fixed top-16.
Next allowed test: Change the mechanism: a calibrated proposal model validated across evolving checkpoints, a soft/set target that tolerates missing the exact argmin, or a source-training gate that uses a materially different target construction rather than static top-k expansion.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-adaptive-proposal-diagnostic.md`
```
