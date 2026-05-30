# Result-boundary critic proposals help only when followed by broad candidate rescoring.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-uncertainty-proposal-diagnostic.md

Summary:

- Extended the result-boundary amortized critic diagnostic with mean/LCB candidate proposals and optional critic ensembles. A pairwise critic trained on only `8` forced scores per train prompt still had weak direct heldout argmin recovery at step `800` (`0.20` single critic, `0.24` four-member ensemble). If the critic only proposes candidates and those candidates are then actually scored, recovery improves: single-critic top-8 proposal recovers the full-enum best on `0.79` of heldout prompts and top-16 reaches `0.96`; four-member top-8 reaches `0.84` and top-16 reaches `1.00` by mean proposal. But top-16 already scores `16/39 = 41%` of result classes, and the four-member ensemble uses `32` sparse scores per train prompt. LCB uncertainty did not beat the mean proposal (`0.79` vs `0.84` at top-8 step-800 for the ensemble; `0.98` vs `1.00` at top-16). This is a useful candidate-rescoring diagnostic, not a solved scalable/non-prescriptive training method.

Questions:

- What did we learn about Result-boundary critic proposals help only when followed by broad candidate rescoring?
- Has Result-boundary critic proposals help only when followed by broad candidate rescoring been tested?
- Should we repeat Result-boundary critic proposals help only when followed by broad candidate rescoring?
- What is the status of Result-boundary critic proposals help only when followed by broad candidate rescoring?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-uncertainty-proposal-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not claim hidden/output critic argmin is solved, and do not run more beta/ensemble/count tweaks as novelty. The useful finding is that broad proposal-plus-rescoring can approach the full-enum target only at substantial candidate cost; uncertainty LCB did not provide the hoped adaptive-compute advantage.

Next Allowed:

- Change the mechanism before spending training budget: adaptive stopping/calibration that expands only uncertain prompts, soft/set targets that tolerate missing the exact argmin, or a streaming/evolving-checkpoint proposal gate that beats this static fixed-grid diagnostic at materially lower scoring cost.

Full Text:

```text
MIXED: Result-boundary critic proposals help only when followed by broad candidate rescoring.
Conclusion: Extended the result-boundary amortized critic diagnostic with mean/LCB candidate proposals and optional critic ensembles. A pairwise critic trained on only `8` forced scores per train prompt still had weak direct heldout argmin recovery at step `800` (`0.20` single critic, `0.24` four-member ensemble). If the critic only proposes candidates and those candidates are then actually scored, recovery improves: single-critic top-8 proposal recovers the full-enum best on `0.79` of heldout prompts and top-16 reaches `0.96`; four-member top-8 reaches `0.84` and top-16 reaches `1.00` by mean proposal. But top-16 already scores `16/39 = 41%` of result classes, and the four-member ensemble uses `32` sparse scores per train prompt. LCB uncertainty did not beat the mean proposal (`0.79` vs `0.84` at top-8 step-800 for the ensemble; `0.98` vs `1.00` at top-16). This is a useful candidate-rescoring diagnostic, not a solved scalable/non-prescriptive training method.
Do not repeat: Do not claim hidden/output critic argmin is solved, and do not run more beta/ensemble/count tweaks as novelty. The useful finding is that broad proposal-plus-rescoring can approach the full-enum target only at substantial candidate cost; uncertainty LCB did not provide the hoped adaptive-compute advantage.
Next allowed test: Change the mechanism before spending training budget: adaptive stopping/calibration that expands only uncertain prompts, soft/set targets that tolerate missing the exact argmin, or a streaming/evolving-checkpoint proposal gate that beats this static fixed-grid diagnostic at materially lower scoring cost.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-uncertainty-proposal-diagnostic.md`
```
