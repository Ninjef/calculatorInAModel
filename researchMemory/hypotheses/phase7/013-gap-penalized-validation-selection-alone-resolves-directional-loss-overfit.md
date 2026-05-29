# Gap-penalized validation selection alone resolves directional-loss overfit.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-gap-penalized-selection-gate.md

Summary:

- Gap penalties moved `cosine` h16 earlier, but penalty `4` still had result gap `0.1673`, while penalty `5` reduced gap to `0.1511/0.1220` and dropped heldout to `0.6872/0.6979`.

Questions:

- What did we learn about Gap-penalized validation selection alone resolves directional-loss overfit?
- Has Gap-penalized validation selection alone resolves directional-loss overfit been tested?
- Should we repeat Gap-penalized validation selection alone resolves directional-loss overfit?
- What is the status of Gap-penalized validation selection alone resolves directional-loss overfit?
- Why did Gap-penalized validation selection alone resolves directional-loss overfit fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-gap-penalized-selection-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Gap-penalized selection on the same directional-loss `injection_grad_logits`, target-normalized h16/h32 setup with penalties `1/3/4/5`.

Next Allowed:

- Use training-time regularization, target stabilization, or a different learned-gradient state, not checkpoint selection alone.

Full Text:

```text
DISPROVEN: Gap-penalized validation selection alone resolves directional-loss overfit.
Conclusion: Gap penalties moved `cosine` h16 earlier, but penalty `4` still had result gap `0.1673`, while penalty `5` reduced gap to `0.1511/0.1220` and dropped heldout to `0.6872/0.6979`.
Do not repeat: Gap-penalized selection on the same directional-loss `injection_grad_logits`, target-normalized h16/h32 setup with penalties `1/3/4/5`.
Next allowed test: Use training-time regularization, target stabilization, or a different learned-gradient state, not checkpoint selection alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-gap-penalized-selection-gate.md`
```
