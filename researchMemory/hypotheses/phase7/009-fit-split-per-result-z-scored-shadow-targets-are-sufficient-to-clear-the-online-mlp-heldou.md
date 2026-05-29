# Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-normalization-gate.md

Summary:

- Target normalization improved heldout cosines, but `h64/h32/h16` still missed the train-heldout gap gate; best near miss was `h16` with heldout `0.7259/0.7549` and gaps `0.1723/0.1458`.

Questions:

- What did we learn about Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate?
- Has Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate been tested?
- Should we repeat Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate?
- What is the status of Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate?
- Why did Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-normalization-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same per-result z-score target-normalized `h64/h32/h16/h8`, `lr=1e-3`, `100`-step validation-selected Stage 0B sweep as novelty.

Next Allowed:

- Change the shadow input/state or objective more substantially, e.g. richer policy features, explicit regularization, a different loss, or a more stable target construction.

Full Text:

```text
DISPROVEN: Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate.
Conclusion: Target normalization improved heldout cosines, but `h64/h32/h16` still missed the train-heldout gap gate; best near miss was `h16` with heldout `0.7259/0.7549` and gaps `0.1723/0.1458`.
Do not repeat: The same per-result z-score target-normalized `h64/h32/h16/h8`, `lr=1e-3`, `100`-step validation-selected Stage 0B sweep as novelty.
Next allowed test: Change the shadow input/state or objective more substantially, e.g. richer policy features, explicit regularization, a different loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-normalization-gate.md`
```
