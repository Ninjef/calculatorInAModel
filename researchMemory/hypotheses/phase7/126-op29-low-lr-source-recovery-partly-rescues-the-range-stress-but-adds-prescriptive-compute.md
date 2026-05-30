# op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-op29-low-lr-source-recovery-diagnostic.md

Summary:

- Continuing the failed op29 product source step-630 checkpoint for `90` low-LR recovery steps (`lr=0.0003`, one-negative forced-margin weight `0.1`, source stabilization retained) raised source calc from `0.6889` to `0.8211` and final source eval to `0.8233`. The trusted 600-step frozen-policy additive handoff from recovered step `90` reached `0.9067` final eval / `0.8978` step-600 normal, with low controls (`0.0122` injection-zero, `0.0111` forced-random at step `600`) and learned calc `0.8233`. This shows the op29 miss was partly source-maturity limited, but the rescue adds another prescriptive full-grid source continuation and still does not make the method scalable.

Questions:

- What did we learn about op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute?
- Has op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute been tested?
- Should we repeat op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute?
- What is the status of op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute?
- Why did op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute fail?
- What follow-up is allowed for op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-op29-low-lr-source-recovery-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same op29 step-630 to low-LR step-90 recovery and handoff as novelty, and do not extend the same continuation ladder unless explicitly diagnosing a new capacity/recovery hypothesis.

Next Allowed:

- Further range work should change source acquisition, reduce assignment cost against an exact-grid ceiling, or test a materially different source-capacity/recovery mechanism rather than more of the same low-LR recovery.

Full Text:

```text
MIXED-POSITIVE: op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute.
Conclusion: Continuing the failed op29 product source step-630 checkpoint for `90` low-LR recovery steps (`lr=0.0003`, one-negative forced-margin weight `0.1`, source stabilization retained) raised source calc from `0.6889` to `0.8211` and final source eval to `0.8233`. The trusted 600-step frozen-policy additive handoff from recovered step `90` reached `0.9067` final eval / `0.8978` step-600 normal, with low controls (`0.0122` injection-zero, `0.0111` forced-random at step `600`) and learned calc `0.8233`. This shows the op29 miss was partly source-maturity limited, but the rescue adds another prescriptive full-grid source continuation and still does not make the method scalable.
Do not repeat: Do not rerun the same op29 step-630 to low-LR step-90 recovery and handoff as novelty, and do not extend the same continuation ladder unless explicitly diagnosing a new capacity/recovery hypothesis.
Next allowed test: Further range work should change source acquisition, reduce assignment cost against an exact-grid ceiling, or test a materially different source-capacity/recovery mechanism rather than more of the same low-LR recovery.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-low-lr-source-recovery-diagnostic.md`
```
