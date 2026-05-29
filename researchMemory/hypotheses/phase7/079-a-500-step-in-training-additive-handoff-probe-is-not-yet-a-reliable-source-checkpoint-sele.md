# A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md

Summary:

- On fresh no-decay source seed `11`, in-training probe normal @500 chose source step `400` (`0.5625`) over step `800` (`0.5525`), but standalone 600-step frozen-policy handoff favored step `800` by a wide margin (`0.6925` snapshot, `0.7075` final eval) over step `400` (`0.5975` snapshot, `0.6050` final eval).

Questions:

- What did we learn about A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector?
- Has A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector been tested?
- Should we repeat A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector?
- What is the status of A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector?
- Why did A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same source step `400` vs step `800`, 500-step embedded probe plus standalone 600-step verification as novelty.

Next Allowed:

- Treat embedded 500-step probes as logging/triage only, verify with standalone 600-step handoffs, or run embedded probes with 600 steps / richer trend metrics before using them for selection.

Full Text:

```text
MIXED-NEGATIVE: A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector.
Conclusion: On fresh no-decay source seed `11`, in-training probe normal @500 chose source step `400` (`0.5625`) over step `800` (`0.5525`), but standalone 600-step frozen-policy handoff favored step `800` by a wide margin (`0.6925` snapshot, `0.7075` final eval) over step `400` (`0.5975` snapshot, `0.6050` final eval).
Do not repeat: Same source step `400` vs step `800`, 500-step embedded probe plus standalone 600-step verification as novelty.
Next allowed test: Treat embedded 500-step probes as logging/triage only, verify with standalone 600-step handoffs, or run embedded probes with 600 steps / richer trend metrics before using them for selection.
Source: `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`
```
