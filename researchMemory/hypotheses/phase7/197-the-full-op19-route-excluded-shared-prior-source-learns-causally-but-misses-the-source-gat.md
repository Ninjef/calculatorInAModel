# The full op19 route-excluded shared-prior source learns causally but misses the source gate.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op19-route-excluded-shared-prior-source.md

Summary:

- Ran the full op19 four-hook shared-output source with prompt-keyed hard memory, route 1 excluded from direct target discovery, numeric prior replay, full-memory prior fitting every 2 steps, and train-convergence patience 100. The source reached final eval exact/calc `315/400 = 0.7875` and best snapshot `0.8075`, with causal controls low (`0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random). Prompt memory filled only score-eligible routes (`223/223` entries), the excluded/update fraction was `0.3125`, and online prior replay ran `2501` updates after `37,896` forced evals. Train prompts reached `0.840625`, but heldout prompts reached only `0.5625`; prior train/heldout were `0.7781`/`0.5625`. Excluded route 1 was not dead: snapshot route-1 exact/calc was `0.7304`, diagnostic route-1 was `0.8000`, and heldout route-1 was `0.7391`. This confirms shared numeric prior pressure can train a route with no direct prompt-memory updates better than the op9 preflight, but it does not clear the trusted source gate. No handoff was run.

Questions:

- What did we learn about The full op19 route-excluded shared-prior source learns causally but misses the source gate?
- Has The full op19 route-excluded shared-prior source learns causally but misses the source gate been tested?
- Should we repeat The full op19 route-excluded shared-prior source learns causally but misses the source gate?
- What is the status of The full op19 route-excluded shared-prior source learns causally but misses the source gate?
- What follow-up is allowed for The full op19 route-excluded shared-prior source learns causally but misses the source gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op19-route-excluded-shared-prior-source.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun this exact op19 route-excluded 5000-step recipe, the short op9 preflights, route-heldout diagnostic ladders, or prior cadence/patience variants as novelty.

Next Allowed:

- Use a stronger explicit shared/global prior objective, route-balanced/global prior replay, shared target discovery across calculators, or a credit mechanism that removes per-route prompt-memory target tables and answer-derived candidate scoring.

Full Text:

```text
MIXED-POSITIVE: The full op19 route-excluded shared-prior source learns causally but misses the source gate.
Conclusion: Ran the full op19 four-hook shared-output source with prompt-keyed hard memory, route 1 excluded from direct target discovery, numeric prior replay, full-memory prior fitting every 2 steps, and train-convergence patience 100. The source reached final eval exact/calc `315/400 = 0.7875` and best snapshot `0.8075`, with causal controls low (`0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random). Prompt memory filled only score-eligible routes (`223/223` entries), the excluded/update fraction was `0.3125`, and online prior replay ran `2501` updates after `37,896` forced evals. Train prompts reached `0.840625`, but heldout prompts reached only `0.5625`; prior train/heldout were `0.7781`/`0.5625`. Excluded route 1 was not dead: snapshot route-1 exact/calc was `0.7304`, diagnostic route-1 was `0.8000`, and heldout route-1 was `0.7391`. This confirms shared numeric prior pressure can train a route with no direct prompt-memory updates better than the op9 preflight, but it does not clear the trusted source gate. No handoff was run.
Do not repeat: Do not rerun this exact op19 route-excluded 5000-step recipe, the short op9 preflights, route-heldout diagnostic ladders, or prior cadence/patience variants as novelty.
Next allowed test: Use a stronger explicit shared/global prior objective, route-balanced/global prior replay, shared target discovery across calculators, or a credit mechanism that removes per-route prompt-memory target tables and answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op19-route-excluded-shared-prior-source.md`
```
