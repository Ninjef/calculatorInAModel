# Validation-selected early stopping is enough to rescue the simple online MLP shadow module.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gate.md

Summary:

- With `h64`, `lr=1e-3`, `100` steps, `0.1` validation and `0.2` heldout test, the selected step `60` reached test result/upstream cosines `0.6449/0.7266` with train-test gaps `0.3201/0.2414`.

Questions:

- What did we learn about Validation-selected early stopping is enough to rescue the simple online MLP shadow module?
- Has Validation-selected early stopping is enough to rescue the simple online MLP shadow module been tested?
- Should we repeat Validation-selected early stopping is enough to rescue the simple online MLP shadow module?
- What is the status of Validation-selected early stopping is enough to rescue the simple online MLP shadow module?
- Why did Validation-selected early stopping is enough to rescue the simple online MLP shadow module fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Treating validation-best checkpoints from this same simple MLP as a Stage 1 go signal.

Next Allowed:

- Change the learned-gradient target or state itself, such as target normalization, regularization, richer policy features, or a different synthetic-gradient objective.

Full Text:

```text
DISPROVEN: Validation-selected early stopping is enough to rescue the simple online MLP shadow module.
Conclusion: With `h64`, `lr=1e-3`, `100` steps, `0.1` validation and `0.2` heldout test, the selected step `60` reached test result/upstream cosines `0.6449/0.7266` with train-test gaps `0.3201/0.2414`.
Do not repeat: Treating validation-best checkpoints from this same simple MLP as a Stage 1 go signal.
Next allowed test: Change the learned-gradient target or state itself, such as target normalization, regularization, richer policy features, or a different synthetic-gradient objective.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gate.md`
```
