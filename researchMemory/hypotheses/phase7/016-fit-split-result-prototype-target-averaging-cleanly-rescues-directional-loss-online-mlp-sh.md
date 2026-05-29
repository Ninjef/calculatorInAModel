# Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-prototype-gate.md

Summary:

- Prototype targets slightly improved the tradeoff but not enough; h32/cosine reached heldout `0.8040/0.8243` with gaps `0.1909/0.1557`, and h16/cosine plus gap selection reached `0.7540/0.7855` with gaps `0.1705/0.1409`.

Questions:

- What did we learn about Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit?
- Has Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit been tested?
- Should we repeat Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit?
- What is the status of Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit?
- Why did Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-prototype-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- `fit_result_prototype` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup, including gap penalties `3/4/5`, as novelty.

Next Allowed:

- Different learned-gradient state, explicit train-time gap/norm penalties, or a target construction richer than boundary-best class prototypes.

Full Text:

```text
DISPROVEN: Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit.
Conclusion: Prototype targets slightly improved the tradeoff but not enough; h32/cosine reached heldout `0.8040/0.8243` with gaps `0.1909/0.1557`, and h16/cosine plus gap selection reached `0.7540/0.7855` with gaps `0.1705/0.1409`.
Do not repeat: `fit_result_prototype` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup, including gap penalties `3/4/5`, as novelty.
Next allowed test: Different learned-gradient state, explicit train-time gap/norm penalties, or a target construction richer than boundary-best class prototypes.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-prototype-gate.md`
```
