# Candidate-evidence prior updates are wired for the next route-excluded source gate.

Kind: hypothesis_memory
Status: TOOLING
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-prior-tooling.md

Summary:

- Added `--result-boundary-target-amortized-prior-candidate-evidence-weight`, which trains the shared amortized prior directly on already-scored positive candidate targets during prompt-memory target discovery, before prompt memory freezes. A tiny op2 route-excluded smoke produced `27` candidate-evidence prior updates over `81` examples and confirmed the metrics path. This is not source-quality evidence and no handoff was run.

Questions:

- What did we learn about Candidate-evidence prior updates are wired for the next route-excluded source gate?
- Has Candidate-evidence prior updates are wired for the next route-excluded source gate been tested?
- Should we repeat Candidate-evidence prior updates are wired for the next route-excluded source gate?
- What is the status of Candidate-evidence prior updates are wired for the next route-excluded source gate?
- What follow-up is allowed for Candidate-evidence prior updates are wired for the next route-excluded source gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-prior-tooling.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat the tiny smoke as evidence that candidate-evidence prior updates improve route-excluded source quality, and do not run route-weight/bootstrap threshold ladders instead of the new source gate.

Next Allowed:

- Run a real op19 route-excluded source with candidate-evidence prior updates enabled, then require heldout prompt quality and excluded-route quality before any trusted handoff.

Full Text:

```text
TOOLING: Candidate-evidence prior updates are wired for the next route-excluded source gate.
Conclusion: Added `--result-boundary-target-amortized-prior-candidate-evidence-weight`, which trains the shared amortized prior directly on already-scored positive candidate targets during prompt-memory target discovery, before prompt memory freezes. A tiny op2 route-excluded smoke produced `27` candidate-evidence prior updates over `81` examples and confirmed the metrics path. This is not source-quality evidence and no handoff was run.
Do not repeat: Do not treat the tiny smoke as evidence that candidate-evidence prior updates improve route-excluded source quality, and do not run route-weight/bootstrap threshold ladders instead of the new source gate.
Next allowed test: Run a real op19 route-excluded source with candidate-evidence prior updates enabled, then require heldout prompt quality and excluded-route quality before any trusted handoff.
Source: `aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-prior-tooling.md`
```
