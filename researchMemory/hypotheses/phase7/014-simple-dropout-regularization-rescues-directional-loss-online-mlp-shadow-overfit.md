# Simple dropout regularization rescues directional-loss online MLP shadow overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-dropout-regularization-gate.md

Summary:

- Dropout `0.1/0.2` with `weight_decay=0.01` preserved heldout cosines on the target-normalized `cosine` branch, but h16/h32 still had result train-heldout gaps near `0.20`; best h32/dropout `0.1` reached heldout `0.7920/0.8248` with gaps `0.2039/0.1564`.

Questions:

- What did we learn about Simple dropout regularization rescues directional-loss online MLP shadow overfit?
- Has Simple dropout regularization rescues directional-loss online MLP shadow overfit been tested?
- Should we repeat Simple dropout regularization rescues directional-loss online MLP shadow overfit?
- What is the status of Simple dropout regularization rescues directional-loss online MLP shadow overfit?
- Why did Simple dropout regularization rescues directional-loss online MLP shadow overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-dropout-regularization-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Dropout-only h16/h32 sweeps on the same `injection_grad_logits`, target-normalized, `cosine`, `lr=1e-3`, `100`-step setup as novelty.

Next Allowed:

- Change target construction or learned-gradient state, or add explicit training-time gap/norm penalties rather than ordinary dropout alone.

Full Text:

```text
DISPROVEN: Simple dropout regularization rescues directional-loss online MLP shadow overfit.
Conclusion: Dropout `0.1/0.2` with `weight_decay=0.01` preserved heldout cosines on the target-normalized `cosine` branch, but h16/h32 still had result train-heldout gaps near `0.20`; best h32/dropout `0.1` reached heldout `0.7920/0.8248` with gaps `0.2039/0.1564`.
Do not repeat: Dropout-only h16/h32 sweeps on the same `injection_grad_logits`, target-normalized, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: Change target construction or learned-gradient state, or add explicit training-time gap/norm penalties rather than ordinary dropout alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-dropout-regularization-gate.md`
```
