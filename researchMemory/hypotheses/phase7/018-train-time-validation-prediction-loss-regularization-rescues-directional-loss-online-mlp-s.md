# Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-loss-gate.md

Summary:

- h32 with validation-loss weight `0.5/1.0` kept heldout cosines high (`0.7953/0.8233`, `0.7915/0.8195`) but result gaps stayed near `0.199`; h16/weight `1.0` reduced gaps to `0.1595/0.1150` but dropped heldout to `0.7274/0.7381` and inflated norms.

Questions:

- What did we learn about Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit?
- Has Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit been tested?
- Should we repeat Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit?
- What is the status of Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit?
- Why did Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-loss-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Validation-loss weights `0.5/1.0` on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`, `lr=1e-3`, `100`-step setup as novelty.

Next Allowed:

- A direct split-gradient gap/norm objective, Jacobian-conditioned state, or a richer target construction.

Full Text:

```text
DISPROVEN: Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit.
Conclusion: h32 with validation-loss weight `0.5/1.0` kept heldout cosines high (`0.7953/0.8233`, `0.7915/0.8195`) but result gaps stayed near `0.199`; h16/weight `1.0` reduced gaps to `0.1595/0.1150` but dropped heldout to `0.7274/0.7381` and inflated norms.
Do not repeat: Validation-loss weights `0.5/1.0` on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: A direct split-gradient gap/norm objective, Jacobian-conditioned state, or a richer target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-loss-gate.md`
```
