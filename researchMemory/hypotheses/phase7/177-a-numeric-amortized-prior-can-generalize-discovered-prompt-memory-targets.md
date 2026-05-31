# A numeric amortized prior can generalize discovered prompt-memory targets.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md

Summary:

- Added an operand-conditioned amortized prior trained only from prompt hard-memory entries, plus heldout replay hooks and trace/replay diagnostics. On the prior heldout-failed op19 source, the arbitrary embedding prior fit train memory (`1.000` memory fit, `0.9969` train-vs-true) but got `0.0000` heldout-vs-true, confirming it memorizes prompt keys. Switching the prior to normalized numeric operand features kept train fit (`1.000`) and reached `0.9125` heldout-vs-true on the same `80` heldout prompts. A post-hoc result-head replay gate then transferred those numeric pseudo-targets into the source model: heldout calc/exact moved from `0.0875` to `0.9125` while train stayed `0.990625`. The matched embedding-prior replay control ended at only `0.0125` heldout and `0.959375` train.

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

- Do not treat embedding-feature amortized priors as a fresh-prompt solution, and do not claim the numeric prior has solved from-scratch/end-to-end training yet. The positive is target-prior generalization plus post-hoc result-head uptake.

Next Allowed:

- Run the integrated numeric-prior replay streaming source gate, then test whether the source learns seen and heldout prompts without post-hoc replay; only after that consider trusted handoff.

Full Text:

```text
PARTIAL-POSITIVE: A numeric amortized prior can generalize discovered prompt-memory targets.
Conclusion: Added an operand-conditioned amortized prior trained only from prompt hard-memory entries, plus heldout replay hooks and trace/replay diagnostics. On the prior heldout-failed op19 source, the arbitrary embedding prior fit train memory (`1.000` memory fit, `0.9969` train-vs-true) but got `0.0000` heldout-vs-true, confirming it memorizes prompt keys. Switching the prior to normalized numeric operand features kept train fit (`1.000`) and reached `0.9125` heldout-vs-true on the same `80` heldout prompts. A post-hoc result-head replay gate then transferred those numeric pseudo-targets into the source model: heldout calc/exact moved from `0.0875` to `0.9125` while train stayed `0.990625`. The matched embedding-prior replay control ended at only `0.0125` heldout and `0.959375` train.
Do not repeat: Do not treat embedding-feature amortized priors as a fresh-prompt solution, and do not claim the numeric prior has solved from-scratch/end-to-end training yet. The positive is target-prior generalization plus post-hoc result-head uptake.
Next allowed test: Run the integrated numeric-prior replay streaming source gate, then test whether the source learns seen and heldout prompts without post-hoc replay; only after that consider trusted handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md`
```
