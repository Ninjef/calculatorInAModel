# Highest source normal/calculator accuracy is a reliable source checkpoint selector.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-source-selection-metric-replication.md

Summary:

- Reproduced `src2` with checkpoints; source step `1300` had higher source normal/calc (`0.9475`) than final (`0.9150`) but transferred worse into additive seed `4` (`0.8675` vs final-control `0.9525`).

Questions:

- What did we learn about Highest source normal/calculator accuracy is a reliable source checkpoint selector?
- Has Highest source normal/calculator accuracy is a reliable source checkpoint selector been tested?
- Should we repeat Highest source normal/calculator accuracy is a reliable source checkpoint selector?
- What is the status of Highest source normal/calculator accuracy is a reliable source checkpoint selector?
- Why did Highest source normal/calculator accuracy is a reliable source checkpoint selector fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-selection-metric-replication.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src2` step-1300 versus final additive seed-4 frozen-policy 800-step transfer as novelty.

Next Allowed:

- Source-quality probes for handoff geometry, source acquisition optimized for transfer/readout learnability, or selected-source replication with a selector beyond source accuracy.

Full Text:

```text
DISPROVEN: Highest source normal/calculator accuracy is a reliable source checkpoint selector.
Conclusion: Reproduced `src2` with checkpoints; source step `1300` had higher source normal/calc (`0.9475`) than final (`0.9150`) but transferred worse into additive seed `4` (`0.8675` vs final-control `0.9525`).
Do not repeat: Same `src2` step-1300 versus final additive seed-4 frozen-policy 800-step transfer as novelty.
Next allowed test: Source-quality probes for handoff geometry, source acquisition optimized for transfer/readout learnability, or selected-source replication with a selector beyond source accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-selection-metric-replication.md`
```
