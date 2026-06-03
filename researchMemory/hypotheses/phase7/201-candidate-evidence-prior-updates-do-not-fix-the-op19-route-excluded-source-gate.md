# Candidate-evidence prior updates do not fix the op19 route-excluded source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-route-excluded-source.md

Summary:

- Ran the full op19 four-hook shared-output route-excluded source with `--result-boundary-target-amortized-prior-candidate-evidence-weight 1.0`. The run reused already-scored positive candidate targets and recorded `32` candidate-evidence prior updates over `1060` examples, but final eval reached only `309/400 = 0.7725` and best/final snapshot normal was `0.8000`, with final controls `0.0475` injection-zero, `0.0025` forced-zero, and `0.0025` forced-random. Train prompts reached `0.80625`, heldout prompts only `0.5375`, and prior train/heldout accuracy was `0.7156`/`0.5375`. Excluded route 1 was not rescued: train route 1 `0.6495`, heldout route 1 `0.6522`, diagnostic route 1 `0.7429`. No handoff was run.

Questions:

- What did we learn about Candidate-evidence prior updates do not fix the op19 route-excluded source gate?
- Has Candidate-evidence prior updates do not fix the op19 route-excluded source gate been tested?
- Should we repeat Candidate-evidence prior updates do not fix the op19 route-excluded source gate?
- What is the status of Candidate-evidence prior updates do not fix the op19 route-excluded source gate?
- Why did Candidate-evidence prior updates do not fix the op19 route-excluded source gate fail?
- What follow-up is allowed for Candidate-evidence prior updates do not fix the op19 route-excluded source gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-route-excluded-source.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run candidate-evidence weight, same-seed, or timing ladders as novelty. The evidence target path fired, but prompt memory filled by step `50`, candidate-evidence updates were only `32`, and the live prior still generalized poorly.

Next Allowed:

- Leave the route-excluded tweak branch and move to a stronger shared/global target mechanism, joint target formation across routes, or a less-prescriptive credit signal that removes per-route prompt-memory target tables and answer-derived candidate scoring.

Full Text:

```text
MIXED-NEGATIVE: Candidate-evidence prior updates do not fix the op19 route-excluded source gate.
Conclusion: Ran the full op19 four-hook shared-output route-excluded source with `--result-boundary-target-amortized-prior-candidate-evidence-weight 1.0`. The run reused already-scored positive candidate targets and recorded `32` candidate-evidence prior updates over `1060` examples, but final eval reached only `309/400 = 0.7725` and best/final snapshot normal was `0.8000`, with final controls `0.0475` injection-zero, `0.0025` forced-zero, and `0.0025` forced-random. Train prompts reached `0.80625`, heldout prompts only `0.5375`, and prior train/heldout accuracy was `0.7156`/`0.5375`. Excluded route 1 was not rescued: train route 1 `0.6495`, heldout route 1 `0.6522`, diagnostic route 1 `0.7429`. No handoff was run.
Do not repeat: Do not run candidate-evidence weight, same-seed, or timing ladders as novelty. The evidence target path fired, but prompt memory filled by step `50`, candidate-evidence updates were only `32`, and the live prior still generalized poorly.
Next allowed test: Leave the route-excluded tweak branch and move to a stronger shared/global target mechanism, joint target formation across routes, or a less-prescriptive credit signal that removes per-route prompt-memory target tables and answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-03-candidate-evidence-route-excluded-source.md`
```
