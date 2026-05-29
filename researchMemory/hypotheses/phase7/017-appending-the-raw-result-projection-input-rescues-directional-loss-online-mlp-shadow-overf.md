# Appending the raw result-projection input rescues directional-loss online MLP shadow overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-result-input-state-gate.md

Summary:

- The result-input state improved upstream heldout alignment, but result gaps remained high; h16/cosine reached heldout `0.7676/0.8372` with gaps `0.1958/0.1269`, and h32/cosine reached `0.7895/0.8294` with gaps `0.2079/0.1533`.

Questions:

- What did we learn about Appending the raw result-projection input rescues directional-loss online MLP shadow overfit?
- Has Appending the raw result-projection input rescues directional-loss online MLP shadow overfit been tested?
- Should we repeat Appending the raw result-projection input rescues directional-loss online MLP shadow overfit?
- What is the status of Appending the raw result-projection input rescues directional-loss online MLP shadow overfit?
- Why did Appending the raw result-projection input rescues directional-loss online MLP shadow overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-result-input-state-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- `injection_grad_logits_result_input` with target z-score, h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100` steps, including h16 gap penalties `3/4/5`, as novelty.

Next Allowed:

- Explicit train-time gap/norm penalties, Jacobian-conditioned state, or another genuinely different learned-gradient target/state.

Full Text:

```text
DISPROVEN: Appending the raw result-projection input rescues directional-loss online MLP shadow overfit.
Conclusion: The result-input state improved upstream heldout alignment, but result gaps remained high; h16/cosine reached heldout `0.7676/0.8372` with gaps `0.1958/0.1269`, and h32/cosine reached `0.7895/0.8294` with gaps `0.2079/0.1533`.
Do not repeat: `injection_grad_logits_result_input` with target z-score, h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100` steps, including h16 gap penalties `3/4/5`, as novelty.
Next allowed test: Explicit train-time gap/norm penalties, Jacobian-conditioned state, or another genuinely different learned-gradient target/state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-result-input-state-gate.md`
```
