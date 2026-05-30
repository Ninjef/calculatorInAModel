# Longer frozen-upstream routed source did not prove leakage returns.

Kind: hypothesis_memory
Status: SUPERSEDED-MEASUREMENT-BUG
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md

Summary:

- Extended the fair `embd32` two-hook routed source with cloned output projections and frozen upstream from `200` to `630` steps. The longer run recovered source learning: final eval reached `379/400 = 0.9475`, the last step-630 snapshot reached `0.9750` normal, and both active hooks trained (`0.9955/0.9494` hook calc on the 400-sample snapshot; `0.9286/0.9444` in the final 128-sample routed summary). But the causal control rose with learning: step-630 injection-zero was `0.4400`, and final 128-sample counterfactual injection-zero was `0.5000`, close to the open-upstream routed source leak (`0.4600` snapshot, `0.53125` final). The earlier source200 frozen-upstream run was low-leak mainly because it was undertrained, not because this recipe had solved routed causal acquisition.

Questions:

- What did we learn about Longer frozen-upstream routed source did not prove leakage returns?
- Has Longer frozen-upstream routed source did not prove leakage returns been tested?
- Should we repeat Longer frozen-upstream routed source did not prove leakage returns?
- What is the status of Longer frozen-upstream routed source did not prove leakage returns?
- What follow-up is allowed for Longer frozen-upstream routed source did not prove leakage returns?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun frozen-upstream source630 as an anti-leak test unless the question is specifically about frozen-upstream learning speed; the leakage claim was invalidated by the counterfactual-control bug.

Next Allowed:

- Re-evaluate or rerun routed handoff/source claims only with corrected all-hook counterfactuals, then move to fresh-seed/more-hook/shared-output validation.

Full Text:

```text
SUPERSEDED-MEASUREMENT-BUG: Longer frozen-upstream routed source did not prove leakage returns.
Conclusion: Extended the fair `embd32` two-hook routed source with cloned output projections and frozen upstream from `200` to `630` steps. The longer run recovered source learning: final eval reached `379/400 = 0.9475`, the last step-630 snapshot reached `0.9750` normal, and both active hooks trained (`0.9955/0.9494` hook calc on the 400-sample snapshot; `0.9286/0.9444` in the final 128-sample routed summary). But the causal control rose with learning: step-630 injection-zero was `0.4400`, and final 128-sample counterfactual injection-zero was `0.5000`, close to the open-upstream routed source leak (`0.4600` snapshot, `0.53125` final). The earlier source200 frozen-upstream run was low-leak mainly because it was undertrained, not because this recipe had solved routed causal acquisition.
Correction: this conclusion used the same invalid primary-hook-only zeroing helper. It should not be used as evidence that frozen-upstream learning leaks. The broader lesson is instrumentation discipline: routed multi-hook controls need all hooks ablated before interpreting source leakage.
Do not repeat: Do not rerun frozen-upstream source630 as an anti-leak test unless the question is specifically about frozen-upstream learning speed; the leakage claim was invalidated by the counterfactual-control bug.
Next allowed test: Re-evaluate or rerun routed handoff/source claims only with corrected all-hook counterfactuals, then move to fresh-seed/more-hook/shared-output validation.
Source: `aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md`
```
