# Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-source-checkpoint-selection-gate.md

Summary:

- Reproducing `src5` with checkpoint snapshots and transferring the source step-1500 checkpoint (`0.9200` source normal/calc) improved immediate frozen-policy additive handoff from the old final-checkpoint baseline `0.5550` to `0.6975`.

Questions:

- What did we learn about Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient?
- Has Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient been tested?
- Should we repeat Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient?
- What is the status of Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-source-checkpoint-selection-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same `src5` step-1500 selected-source checkpoint into additive seed `5`, frozen-policy, 800-step handoff as novelty.

Next Allowed:

- Source-selection metrics beyond normal/calc accuracy, source acquisition for handoff-friendly geometry, stronger selected-source replication, or utility-aware stable-policy readout adaptation.

Full Text:

```text
PARTIAL: Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient.
Conclusion: Reproducing `src5` with checkpoint snapshots and transferring the source step-1500 checkpoint (`0.9200` source normal/calc) improved immediate frozen-policy additive handoff from the old final-checkpoint baseline `0.5550` to `0.6975`.
Do not repeat: Same `src5` step-1500 selected-source checkpoint into additive seed `5`, frozen-policy, 800-step handoff as novelty.
Next allowed test: Source-selection metrics beyond normal/calc accuracy, source acquisition for handoff-friendly geometry, stronger selected-source replication, or utility-aware stable-policy readout adaptation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-checkpoint-selection-gate.md`
```
