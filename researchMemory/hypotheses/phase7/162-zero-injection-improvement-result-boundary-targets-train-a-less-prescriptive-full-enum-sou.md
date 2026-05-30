# Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-source-gate.md

Summary:

- Added `result_boundary_target_mode=zero_improvement`, which weights forced result classes by answer-loss improvement over the zero-injection baseline instead of selecting the true sum or argmin directly. In the 200-step upstream-open source gate, full enumeration reached step-200 snapshot calc `0.5700`, learned-best/source calc `0.5475`, and final eval `217/400 = 0.5425`, matching nearby full-enum hard-best comparators while assigning true-result target mass `0.9541` and effective results `1.2692`. The paired topk8+unique24 sparse gate improved over sampled hard-best (`0.4300` final vs `0.3525`) but still missed full-enum zero-improvement (`0.5425`) despite `0.9725` true-candidate coverage.

Questions:

- What did we learn about Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring?
- Has Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring been tested?
- Should we repeat Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring?
- What is the status of Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring?
- What follow-up is allowed for Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat zero-improvement as solved scalability, and do not run blind sample-count ladders. The valuable finding is that a no-calculator utility baseline is a viable less-prescriptive full-enum target and a better sparse target than sampled hard-best, but sparse scoring still needs a stronger proposal/training mechanism.

Next Allowed:

- Continue only with a high-leverage scaling step: longer source/handoff validation for full-enum zero-improvement, or an active proposal/streaming mechanism that closes the sampled gap at materially lower scoring cost.

Full Text:

```text
PARTIAL-POSITIVE: Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring.
Conclusion: Added `result_boundary_target_mode=zero_improvement`, which weights forced result classes by answer-loss improvement over the zero-injection baseline instead of selecting the true sum or argmin directly. In the 200-step upstream-open source gate, full enumeration reached step-200 snapshot calc `0.5700`, learned-best/source calc `0.5475`, and final eval `217/400 = 0.5425`, matching nearby full-enum hard-best comparators while assigning true-result target mass `0.9541` and effective results `1.2692`. The paired topk8+unique24 sparse gate improved over sampled hard-best (`0.4300` final vs `0.3525`) but still missed full-enum zero-improvement (`0.5425`) despite `0.9725` true-candidate coverage.
Do not repeat: Do not treat zero-improvement as solved scalability, and do not run blind sample-count ladders. The valuable finding is that a no-calculator utility baseline is a viable less-prescriptive full-enum target and a better sparse target than sampled hard-best, but sparse scoring still needs a stronger proposal/training mechanism.
Next allowed test: Continue only with a high-leverage scaling step: longer source/handoff validation for full-enum zero-improvement, or an active proposal/streaming mechanism that closes the sampled gap at materially lower scoring cost.
Source: `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-source-gate.md`
```
