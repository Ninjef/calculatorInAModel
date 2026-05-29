# A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md

Summary:

- In a reduced `operand_max=9`, 100-step seed-13 source gate, adding `--additive-forced-true-loss-weight 0.5` made the true result the best forced additive result on `0.5900` of the grid (`top3=0.6900`) versus baseline `0.0000`/`0.0000`, and lowered 50-step additive slope final loss (`0.7367` vs `1.5305`). It also weakened source policy acquisition at the same budget (`0.2800` source calc and `0.2800` final eval vs baseline `0.3500` calc and `0.3800` final eval).

Questions:

- What did we learn about A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition?
- Has A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition been tested?
- Should we repeat A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition?
- What is the status of A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition?
- What follow-up is allowed for A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same small `operand_max=9`, 100-step seed-13 baseline vs aux-weight `0.5` source/geometry gate as novelty.

Next Allowed:

- Use a scheduled/gated auxiliary or retention anchor to avoid competing with source policy acquisition, then verify on `operand_max=19` with targeted standalone 600-step additive handoff gates.

Full Text:

```text
MIXED-POSITIVE: A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition.
Conclusion: In a reduced `operand_max=9`, 100-step seed-13 source gate, adding `--additive-forced-true-loss-weight 0.5` made the true result the best forced additive result on `0.5900` of the grid (`top3=0.6900`) versus baseline `0.0000`/`0.0000`, and lowered 50-step additive slope final loss (`0.7367` vs `1.5305`). It also weakened source policy acquisition at the same budget (`0.2800` source calc and `0.2800` final eval vs baseline `0.3500` calc and `0.3800` final eval).
Do not repeat: The same small `operand_max=9`, 100-step seed-13 baseline vs aux-weight `0.5` source/geometry gate as novelty.
Next allowed test: Use a scheduled/gated auxiliary or retention anchor to avoid competing with source policy acquisition, then verify on `operand_max=19` with targeted standalone 600-step additive handoff gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`
```
