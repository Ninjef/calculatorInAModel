# Longer frozen-upstream routed source learns both hooks but leakage returns.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md

Summary:

- Extended the fair `embd32` two-hook routed source with cloned output projections and frozen upstream from `200` to `630` steps. The longer run recovered source learning: final eval reached `379/400 = 0.9475`, the last step-630 snapshot reached `0.9750` normal, and both active hooks trained (`0.9955/0.9494` hook calc on the 400-sample snapshot; `0.9286/0.9444` in the final 128-sample routed summary). But the causal control rose with learning: step-630 injection-zero was `0.4400`, and final 128-sample counterfactual injection-zero was `0.5000`, close to the open-upstream routed source leak (`0.4600` snapshot, `0.53125` final). The earlier source200 frozen-upstream run was low-leak mainly because it was undertrained, not because this recipe had solved routed causal acquisition.

Questions:

- What did we learn about Longer frozen-upstream routed source learns both hooks but leakage returns?
- Has Longer frozen-upstream routed source learns both hooks but leakage returns been tested?
- Should we repeat Longer frozen-upstream routed source learns both hooks but leakage returns?
- What is the status of Longer frozen-upstream routed source learns both hooks but leakage returns?
- Why did Longer frozen-upstream routed source learns both hooks but leakage returns fail?
- What follow-up is allowed for Longer frozen-upstream routed source learns both hooks but leakage returns?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run the same frozen-upstream `embd32` routed source630 recipe as the anti-leak fix. Longer frozen-upstream training alone is now tested and leaks once the source learns.

Next Allowed:

- Add explicit anti-leak pressure during routed source acquisition, such as source-time injection-zero/causal-gap loss, source ablation controls, stricter bottlenecked route design, or a tied/shared output-projection architecture that is validated by low injection-zero before any handoff.

Full Text:

```text
MIXED-NEGATIVE: Longer frozen-upstream routed source learns both hooks but leakage returns.
Conclusion: Extended the fair `embd32` two-hook routed source with cloned output projections and frozen upstream from `200` to `630` steps. The longer run recovered source learning: final eval reached `379/400 = 0.9475`, the last step-630 snapshot reached `0.9750` normal, and both active hooks trained (`0.9955/0.9494` hook calc on the 400-sample snapshot; `0.9286/0.9444` in the final 128-sample routed summary). But the causal control rose with learning: step-630 injection-zero was `0.4400`, and final 128-sample counterfactual injection-zero was `0.5000`, close to the open-upstream routed source leak (`0.4600` snapshot, `0.53125` final). The earlier source200 frozen-upstream run was low-leak mainly because it was undertrained, not because this recipe had solved routed causal acquisition.
Do not repeat: Do not run the same frozen-upstream `embd32` routed source630 recipe as the anti-leak fix. Longer frozen-upstream training alone is now tested and leaks once the source learns.
Next allowed test: Add explicit anti-leak pressure during routed source acquisition, such as source-time injection-zero/causal-gap loss, source ablation controls, stricter bottlenecked route design, or a tied/shared output-projection architecture that is validated by low injection-zero before any handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md`
```
