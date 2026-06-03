# Background evidence refresh does not fix the op19 route-excluded source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-background-evidence-refresh-route-excluded-source.md

Summary:

- Ran the full op19 four-hook shared-output route-excluded source with background evidence refresh scoring fresh train-pool prompts every `10` steps, batch `32`, weight `1.0`, and route `1` excluded from refresh scoring while prior replay still trained all routes. Refresh fired heavily: `501` refresh/prior-evidence updates over `11,056` examples and `267,216` forced-result evals, in addition to `2,501` prior updates and `42,144` online-memory forced evals. The source missed badly: final eval exact/calc was `252/400 = 0.6300`, best/final snapshot normal `0.6800`/`0.6475`, with final controls `0.0475` injection-zero, `0.0025` forced-zero, and `0.0025` forced-random. Prompt train/heldout were `0.684375`/`0.3625`, prior train/heldout only `0.50625`/`0.3875`, and excluded route 1 was weak (`0.3505` train, `0.5217` heldout, `0.4857` diagnostic). No trusted handoff was run.

Questions:

- What did we learn about Background evidence refresh does not fix the op19 route-excluded source gate?
- Has Background evidence refresh does not fix the op19 route-excluded source gate been tested?
- Should we repeat Background evidence refresh does not fix the op19 route-excluded source gate?
- What is the status of Background evidence refresh does not fix the op19 route-excluded source gate?
- Why did Background evidence refresh does not fix the op19 route-excluded source gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-background-evidence-refresh-route-excluded-source.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run evidence-refresh batch/every/weight/exclude-route ladders as novelty. Fresh background scoring increased candidate cost and degraded prompt/prior generalization; the route-excluded branch is closed.

Next Allowed:

- Move to a genuinely different shared/global target mechanism or a less-prescriptive credit signal that removes per-route prompt-memory target tables and answer-derived candidate scoring.

Full Text:

```text
MIXED-NEGATIVE: Background evidence refresh does not fix the op19 route-excluded source gate.
Conclusion: Ran the full op19 four-hook shared-output route-excluded source with background evidence refresh scoring fresh train-pool prompts every `10` steps, batch `32`, weight `1.0`, and route `1` excluded from refresh scoring while prior replay still trained all routes. Refresh fired heavily: `501` refresh/prior-evidence updates over `11,056` examples and `267,216` forced-result evals, in addition to `2,501` prior updates and `42,144` online-memory forced evals. The source missed badly: final eval exact/calc was `252/400 = 0.6300`, best/final snapshot normal `0.6800`/`0.6475`, with final controls `0.0475` injection-zero, `0.0025` forced-zero, and `0.0025` forced-random. Prompt train/heldout were `0.684375`/`0.3625`, prior train/heldout only `0.50625`/`0.3875`, and excluded route 1 was weak (`0.3505` train, `0.5217` heldout, `0.4857` diagnostic). No trusted handoff was run.
Do not repeat: Do not run evidence-refresh batch/every/weight/exclude-route ladders as novelty. Fresh background scoring increased candidate cost and degraded prompt/prior generalization; the route-excluded branch is closed.
Next allowed test: Move to a genuinely different shared/global target mechanism or a less-prescriptive credit signal that removes per-route prompt-memory target tables and answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-03-background-evidence-refresh-route-excluded-source.md`
```
