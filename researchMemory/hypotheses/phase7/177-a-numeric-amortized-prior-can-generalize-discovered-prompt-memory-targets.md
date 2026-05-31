# A numeric amortized prior can generalize discovered prompt-memory targets.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md

Summary:

- Added an operand-conditioned amortized prior trained only from prompt hard-memory entries, plus heldout replay hooks and a trace diagnostic. On the prior heldout-failed op19 source, the arbitrary embedding prior fit train memory (`1.000` memory fit, `0.9969` train-vs-true) but got `0.0000` heldout-vs-true, confirming it memorizes prompt keys. Switching the prior to normalized numeric operand features kept train fit (`1.000`) and reached `0.9125` heldout-vs-true on the same `80` heldout prompts, using only the discovered train-memory calculator results as labels. Tiny integrated smoke runs verified the prior replay path executes, but no full source run has shown that the model result policy absorbs these pseudo-targets yet.

Questions:

- What did we learn about A numeric amortized prior can generalize discovered prompt-memory targets?
- Has A numeric amortized prior can generalize discovered prompt-memory targets been tested?
- Should we repeat A numeric amortized prior can generalize discovered prompt-memory targets?
- What is the status of A numeric amortized prior can generalize discovered prompt-memory targets?
- Why did A numeric amortized prior can generalize discovered prompt-memory targets fail?
- What follow-up is allowed for A numeric amortized prior can generalize discovered prompt-memory targets?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat embedding-feature amortized priors as a fresh-prompt solution, and do not claim the numeric prior has solved model training until an op19 heldout source gate lifts heldout calculator-result accuracy under prior replay.

Next Allowed:

- Run the full numeric-prior heldout source gate with unscored heldout replay; compare heldout calc/exact against the `0.0875` no-prior boundary and track whether prior accuracy transfers into model result logits.

Full Text:

```text
PARTIAL-POSITIVE: A numeric amortized prior can generalize discovered prompt-memory targets.
Conclusion: Added an operand-conditioned amortized prior trained only from prompt hard-memory entries, plus heldout replay hooks and a trace diagnostic. On the prior heldout-failed op19 source, the arbitrary embedding prior fit train memory (`1.000` memory fit, `0.9969` train-vs-true) but got `0.0000` heldout-vs-true, confirming it memorizes prompt keys. Switching the prior to normalized numeric operand features kept train fit (`1.000`) and reached `0.9125` heldout-vs-true on the same `80` heldout prompts, using only the discovered train-memory calculator results as labels. Tiny integrated smoke runs verified the prior replay path executes, but no full source run has shown that the model result policy absorbs these pseudo-targets yet.
Do not repeat: Do not treat embedding-feature amortized priors as a fresh-prompt solution, and do not claim the numeric prior has solved model training until an op19 heldout source gate lifts heldout calculator-result accuracy under prior replay.
Next allowed test: Run the full numeric-prior heldout source gate with unscored heldout replay; compare heldout calc/exact against the `0.0875` no-prior boundary and track whether prior accuracy transfers into model result logits.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md`
```
