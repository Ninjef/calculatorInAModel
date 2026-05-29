# Additive forced-result geometry can select source checkpoints inside known source families.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md

Summary:

- On known `src2/src4/src5` handoff comparisons, geometry only partially identified `src5` step `1100`; it tied or favored non-winners for `src4` and tied `src2` step `1300` versus final, while true-best gap selected wrong checkpoints.

Questions:

- What did we learn about Additive forced-result geometry can select source checkpoints inside known source families?
- Has Additive forced-result geometry can select source checkpoints inside known source families been tested?
- Should we repeat Additive forced-result geometry can select source checkpoints inside known source families?
- What is the status of Additive forced-result geometry can select source checkpoints inside known source families?
- Why did Additive forced-result geometry can select source checkpoints inside known source families fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same geometry scan over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.

Next Allowed:

- Use geometry only as logging/warning, optimize source acquisition for actual early handoff slope, or design a stronger one/few-update proxy.

Full Text:

```text
DISPROVEN: Additive forced-result geometry can select source checkpoints inside known source families.
Conclusion: On known `src2/src4/src5` handoff comparisons, geometry only partially identified `src5` step `1100`; it tied or favored non-winners for `src4` and tied `src2` step `1300` versus final, while true-best gap selected wrong checkpoints.
Do not repeat: Same geometry scan over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Use geometry only as logging/warning, optimize source acquisition for actual early handoff slope, or design a stronger one/few-update proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`
```
