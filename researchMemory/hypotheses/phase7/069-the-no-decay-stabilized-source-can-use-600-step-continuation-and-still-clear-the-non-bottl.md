# The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-reduced-continuation.md

Summary:

- Reading out from the step-600 continuation checkpoint reached final eval `0.9425` with injection-zero `0.0078`, forced-random `0.0781`, and learned calc `0.8750`; this is only `0.0150` below the 800-continuation readout.

Questions:

- What did we learn about The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate?
- Has The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate been tested?
- Should we repeat The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate?
- What is the status of The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate?
- What follow-up is allowed for The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-reduced-continuation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same no-decay stabilized step-600 continuation checkpoint into 600-step readout as novelty.

Next Allowed:

- Replicate the 600-continuation recipe on another stabilized source, test 500 continuation, or build a proxy for deciding continuation budget.

Full Text:

```text
POSITIVE: The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate.
Conclusion: Reading out from the step-600 continuation checkpoint reached final eval `0.9425` with injection-zero `0.0078`, forced-random `0.0781`, and learned calc `0.8750`; this is only `0.0150` below the 800-continuation readout.
Do not repeat: Same no-decay stabilized step-600 continuation checkpoint into 600-step readout as novelty.
Next allowed test: Replicate the 600-continuation recipe on another stabilized source, test 500 continuation, or build a proxy for deciding continuation budget.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-reduced-continuation.md`
```
