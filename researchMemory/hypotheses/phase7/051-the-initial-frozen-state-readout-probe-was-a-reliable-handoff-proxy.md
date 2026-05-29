# The initial frozen-state readout probe was a reliable handoff proxy.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-frozen-state-readout-probe.md

Summary:

- Reusable script validation exposed that the scratch probe used the wrong `EQ_ID`/leaky position. Correct safe features over five checkpoints had weak correlation with final handoff (`read_pair 0.2118`, `layer2_pair 0.2865`).

Questions:

- What did we learn about The initial frozen-state readout probe was a reliable handoff proxy?
- Has The initial frozen-state readout probe was a reliable handoff proxy been tested?
- Should we repeat The initial frozen-state readout probe was a reliable handoff proxy?
- What is the status of The initial frozen-state readout probe was a reliable handoff proxy?
- Why did The initial frozen-state readout probe was a reliable handoff proxy fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-frozen-state-readout-probe.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same five-checkpoint safe frozen-state readout probe as novelty, or the leaky/wrong-position answer-token probe.

Next Allowed:

- Build a better non-leaky geometry proxy, validate source selectors on unseen checkpoints, or use 400/600-step handoff probes until a cheaper proxy is proven.

Full Text:

```text
DISPROVEN: The initial frozen-state readout probe was a reliable handoff proxy.
Conclusion: Reusable script validation exposed that the scratch probe used the wrong `EQ_ID`/leaky position. Correct safe features over five checkpoints had weak correlation with final handoff (`read_pair 0.2118`, `layer2_pair 0.2865`).
Do not repeat: Same five-checkpoint safe frozen-state readout probe as novelty, or the leaky/wrong-position answer-token probe.
Next allowed test: Build a better non-leaky geometry proxy, validate source selectors on unseen checkpoints, or use 400/600-step handoff probes until a cheaper proxy is proven.
Source: `aiAgentWorkHistory/phase7/2026-05-29-frozen-state-readout-probe.md`
```
