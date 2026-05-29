# Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-directional-loss-gate.md

Summary:

- `cosine` and `mse_plus_cosine` improved heldout cosines for the simple logits state (`h16/h32` around `0.76-0.79` result, `0.80-0.83` upstream), but result train-heldout gaps stayed around `0.20`; h8 missed heldout cosine.

Questions:

- What did we learn about Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate?
- Has Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate been tested?
- Should we repeat Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate?
- What is the status of Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate?
- Why did Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-directional-loss-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Plain `cosine` or `mse_plus_cosine` with `injection_grad_logits`, per-result target z-score, `h8/h16/h32`, `lr=1e-3`, `100` steps as novelty.

Next Allowed:

- Add explicit norm/gap regularization, a more stable target construction, or a qualitatively different learned-gradient state.

Full Text:

```text
DISPROVEN: Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate.
Conclusion: `cosine` and `mse_plus_cosine` improved heldout cosines for the simple logits state (`h16/h32` around `0.76-0.79` result, `0.80-0.83` upstream), but result train-heldout gaps stayed around `0.20`; h8 missed heldout cosine.
Do not repeat: Plain `cosine` or `mse_plus_cosine` with `injection_grad_logits`, per-result target z-score, `h8/h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Add explicit norm/gap regularization, a more stable target construction, or a qualitatively different learned-gradient state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-directional-loss-gate.md`
```
