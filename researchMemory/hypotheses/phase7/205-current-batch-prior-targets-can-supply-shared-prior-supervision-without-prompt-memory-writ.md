# Current-batch prior targets can supply shared-prior supervision without prompt-memory writes.

Kind: hypothesis_memory
Status: TOOLING
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-current-batch-prior-target-tooling.md

Summary:

- Added `--result-boundary-target-amortized-prior-current-batch-*` and `result_boundary_amortized_prior_current_batch_loss`, which applies detached shared-prior pseudo-targets directly to eligible examples in the live training batch, optionally filtered by routed hook id and confidence. This is distinct from route replay because it acts on the current batch, and distinct from prior bootstrap because it never writes prompt-memory entries. A five-step 2-digit answer-decoder smoke confirmed the objective fired on both logged steps with selected route-1 examples, and a focused unit test verifies route filtering plus confidence gating. This is tooling evidence only, not source-quality evidence.

Questions:

- What did we learn about Current-batch prior targets can supply shared-prior supervision without prompt-memory writes?
- Has Current-batch prior targets can supply shared-prior supervision without prompt-memory writes been tested?
- Should we repeat Current-batch prior targets can supply shared-prior supervision without prompt-memory writes?
- What is the status of Current-batch prior targets can supply shared-prior supervision without prompt-memory writes?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-current-batch-prior-target-tooling.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat the smoke as proof that direct current-batch prior targets fix route-excluded source quality, and do not run route-weight/bootstrap/candidate-evidence/refresh ladders under this new name.

Next Allowed:

- Run a real source gate where direct current-batch shared-prior targets replace or reduce per-route prompt-memory target supply, then require heldout prompt and excluded-route quality before any handoff.

Full Text:

```text
TOOLING: Current-batch prior targets can supply shared-prior supervision without prompt-memory writes.
Conclusion: Added `--result-boundary-target-amortized-prior-current-batch-*` and `result_boundary_amortized_prior_current_batch_loss`, which applies detached shared-prior pseudo-targets directly to eligible examples in the live training batch, optionally filtered by routed hook id and confidence. This is distinct from route replay because it acts on the current batch, and distinct from prior bootstrap because it never writes prompt-memory entries. A five-step 2-digit answer-decoder smoke confirmed the objective fired on both logged steps with selected route-1 examples, and a focused unit test verifies route filtering plus confidence gating. This is tooling evidence only, not source-quality evidence.
Do not repeat: Do not treat the smoke as proof that direct current-batch prior targets fix route-excluded source quality, and do not run route-weight/bootstrap/candidate-evidence/refresh ladders under this new name.
Next allowed test: Run a real source gate where direct current-batch shared-prior targets replace or reduce per-route prompt-memory target supply, then require heldout prompt and excluded-route quality before any handoff.
Source: `aiAgentWorkHistory/phase7/2026-06-03-current-batch-prior-target-tooling.md`
```
