# A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-warmup-gate.md

Summary:

- Hidden size `64` reached heldout result/upstream cosines `0.7167/0.7601`, but train-heldout gaps were `0.2683/0.2202`; hidden size `16` reduced the gap but heldout result cosine fell to `0.6255`.

Questions:

- What did we learn about A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate?
- Has A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate been tested?
- Should we repeat A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate?
- What is the status of A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate?
- Why did A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-warmup-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Launching Stage 1 from these simple online-MLP warmups, or rerunning the same `h64`/`h16`, `lr=1e-3`, `100`-step gate as novelty.

Next Allowed:

- Add a genuinely stronger shadow-generalization mechanism, such as validation early stopping, regularization, target normalization, richer policy state, or a different synthetic-gradient objective, and gate it heldout before Stage 1.

Full Text:

```text
DISPROVEN: A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate.
Conclusion: Hidden size `64` reached heldout result/upstream cosines `0.7167/0.7601`, but train-heldout gaps were `0.2683/0.2202`; hidden size `16` reduced the gap but heldout result cosine fell to `0.6255`.
Do not repeat: Launching Stage 1 from these simple online-MLP warmups, or rerunning the same `h64`/`h16`, `lr=1e-3`, `100`-step gate as novelty.
Next allowed test: Add a genuinely stronger shadow-generalization mechanism, such as validation early stopping, regularization, target normalization, richer policy state, or a different synthetic-gradient objective, and gate it heldout before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-warmup-gate.md`
```
