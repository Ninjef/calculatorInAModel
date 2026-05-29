# The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-boundary.md

Summary:

- 600-step readout from continuation checkpoints reached final eval `0.9400` at step `500`, `0.9175` at step `400`, and `0.8850` at step `300`; controls stayed far below normal.

Questions:

- What did we learn about The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval?
- Has The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval been tested?
- Should we repeat The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval?
- What is the status of The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval?
- What follow-up is allowed for The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-boundary.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same no-decay stabilized `600/500/400/300` continuation-checkpoint readout ladder as novelty.

Next Allowed:

- Replicate the 400/500 boundary on another no-decay stabilized source, validate a readout-snapshot selector, or build a cheap continuation-budget proxy.

Full Text:

```text
MIXED-POSITIVE: The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval.
Conclusion: 600-step readout from continuation checkpoints reached final eval `0.9400` at step `500`, `0.9175` at step `400`, and `0.8850` at step `300`; controls stayed far below normal.
Do not repeat: Same no-decay stabilized `600/500/400/300` continuation-checkpoint readout ladder as novelty.
Next allowed test: Replicate the 400/500 boundary on another no-decay stabilized source, validate a readout-snapshot selector, or build a cheap continuation-budget proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-boundary.md`
```
