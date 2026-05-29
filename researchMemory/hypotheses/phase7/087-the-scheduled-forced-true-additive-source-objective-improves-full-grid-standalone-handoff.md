# The scheduled forced-true additive source objective improves full-grid standalone handoff.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md

Summary:

- On `operand_max=19`, seed-13, 200-step source acquisition, scheduled aux (`weight=0.5`, start step `50`) approximately tied baseline source policy (`0.2800` vs `0.2875` train calc; `0.2750` vs `0.2825` final eval) but strongly improved additive geometry (`forced_best_true=0.2125` vs `0.0000`, 50-step slope loss `1.0360` vs `1.8058`) and the trusted 600-step frozen-policy handoff (`0.4150` final eval / `0.3925` step-600 snapshot vs baseline `0.2525` / `0.2625`).

Questions:

- What did we learn about The scheduled forced-true additive source objective improves full-grid standalone handoff?
- Has The scheduled forced-true additive source objective improves full-grid standalone handoff been tested?
- Should we repeat The scheduled forced-true additive source objective improves full-grid standalone handoff?
- What is the status of The scheduled forced-true additive source objective improves full-grid standalone handoff?
- What follow-up is allowed for The scheduled forced-true additive source objective improves full-grid standalone handoff?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-13, `operand_max=19`, 200-step baseline vs scheduled step-50 source gate plus 600-step handoff as novelty.

Next Allowed:

- Extend scheduled source acquisition to longer horizons (`400/600/800`) and verify selected checkpoints with standalone 600-step additive handoff; add a policy-retention anchor if source accuracy drifts.

Full Text:

```text
POSITIVE: The scheduled forced-true additive source objective improves full-grid standalone handoff.
Conclusion: On `operand_max=19`, seed-13, 200-step source acquisition, scheduled aux (`weight=0.5`, start step `50`) approximately tied baseline source policy (`0.2800` vs `0.2875` train calc; `0.2750` vs `0.2825` final eval) but strongly improved additive geometry (`forced_best_true=0.2125` vs `0.0000`, 50-step slope loss `1.0360` vs `1.8058`) and the trusted 600-step frozen-policy handoff (`0.4150` final eval / `0.3925` step-600 snapshot vs baseline `0.2525` / `0.2625`).
Do not repeat: The same seed-13, `operand_max=19`, 200-step baseline vs scheduled step-50 source gate plus 600-step handoff as novelty.
Next allowed test: Extend scheduled source acquisition to longer horizons (`400/600/800`) and verify selected checkpoints with standalone 600-step additive handoff; add a policy-retention anchor if source accuracy drifts.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`
```
