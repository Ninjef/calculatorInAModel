# Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-h128-prior-capacity-stress.md

Summary:

- Re-ran the op29 eval-only target-stratified source with the same constant fit batch `160` and h128 numeric prior instead of h64. The source improved overall exact/calc from `0.9622` to `0.9767` and train exact/calc to `0.9986`, but heldout exact/calc reached only `0.8611`, below the source gate. The online prior was still weak (`0.8097` train / `0.7111` heldout), the validation stop never fired, prior updates stayed at `2501`, and forced-result evals were `294,912`. Post-hoc h128 full-memory fitting from the same trace reached train/heldout `0.9944`/`0.9278`, with train targets matching true sums at `0.9986`. Capacity helps offline but does not repair online constant-batch fit dynamics.

Questions:

- What did we learn about Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting?
- Has Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting been tested?
- Should we repeat Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting?
- What is the status of Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting?
- Why did Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-h128-prior-capacity-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run op29 constant-batch hidden-size bumps or trusted handoff from this heldout-missed source as progress.

Next Allowed:

- Change online fit dynamics directly: e.g. post-memory-fill full-memory refresh, staged full-fit then coreset replay, or coverage-aware/proportional fitting with explicit source-acquisition cost accounting.

Full Text:

```text
MIXED-NEGATIVE: Increasing op29 numeric prior capacity to h128 does not fix online constant-batch fitting.
Conclusion: Re-ran the op29 eval-only target-stratified source with the same constant fit batch `160` and h128 numeric prior instead of h64. The source improved overall exact/calc from `0.9622` to `0.9767` and train exact/calc to `0.9986`, but heldout exact/calc reached only `0.8611`, below the source gate. The online prior was still weak (`0.8097` train / `0.7111` heldout), the validation stop never fired, prior updates stayed at `2501`, and forced-result evals were `294,912`. Post-hoc h128 full-memory fitting from the same trace reached train/heldout `0.9944`/`0.9278`, with train targets matching true sums at `0.9986`. Capacity helps offline but does not repair online constant-batch fit dynamics.
Do not repeat: Do not run op29 constant-batch hidden-size bumps or trusted handoff from this heldout-missed source as progress.
Next allowed test: Change online fit dynamics directly: e.g. post-memory-fill full-memory refresh, staged full-fit then coreset replay, or coverage-aware/proportional fitting with explicit source-acquisition cost accounting.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-h128-prior-capacity-stress.md`
```
