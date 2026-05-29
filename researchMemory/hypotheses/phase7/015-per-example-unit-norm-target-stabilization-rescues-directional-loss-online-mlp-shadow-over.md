# Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-transform-gate.md

Summary:

- Unit-normalizing each target row before fit-split z-scoring preserved the same heldout cosines but kept result gaps near `0.20`; best h32/cosine reached heldout `0.7936/0.8270` with gaps `0.2025/0.1545`.

Questions:

- What did we learn about Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit?
- Has Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit been tested?
- Should we repeat Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit?
- What is the status of Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit?
- Why did Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-transform-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- `unit_norm_per_example` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup as novelty.

Next Allowed:

- More substantial target stabilization, a different learned-gradient state, or explicit train-time gap/norm penalties.

Full Text:

```text
DISPROVEN: Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit.
Conclusion: Unit-normalizing each target row before fit-split z-scoring preserved the same heldout cosines but kept result gaps near `0.20`; best h32/cosine reached heldout `0.7936/0.8270` with gaps `0.2025/0.1545`.
Do not repeat: `unit_norm_per_example` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: More substantial target stabilization, a different learned-gradient state, or explicit train-time gap/norm penalties.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-transform-gate.md`
```
