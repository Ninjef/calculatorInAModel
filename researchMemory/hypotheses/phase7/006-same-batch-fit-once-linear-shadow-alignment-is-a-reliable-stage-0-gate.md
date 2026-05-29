# Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-heldout-linear-shadow-feedback-gate.md

Summary:

- With a deterministic `320/80` split, train result-proj cosine was `0.9981` but heldout result-proj cosine fell to `0.2622`, with a `0.7359` train-heldout gap.

Questions:

- What did we learn about Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate?
- Has Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate been tested?
- Should we repeat Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate?
- What is the status of Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate?
- Why did Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-heldout-linear-shadow-feedback-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Treating same-batch linear shadow alignment as sufficient for training budget.

Next Allowed:

- Online MLP shadow feedback that includes result-policy state and must pass heldout warmup before Stage 1.

Full Text:

```text
DISPROVEN: Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate.
Conclusion: With a deterministic `320/80` split, train result-proj cosine was `0.9981` but heldout result-proj cosine fell to `0.2622`, with a `0.7359` train-heldout gap.
Do not repeat: Treating same-batch linear shadow alignment as sufficient for training budget.
Next allowed test: Online MLP shadow feedback that includes result-policy state and must pass heldout warmup before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-heldout-linear-shadow-feedback-gate.md`
```
