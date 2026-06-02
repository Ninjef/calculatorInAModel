# Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-route-excluded-shared-prior-preflight.md

Summary:

- A follow-up audit found the initial route-exclusion commit passed `memory_update_exclude_routes` to the fixed-batch online-memory branch, not the prompt-keyed branch used by the shared-prior source recipe. Moving it to `result_boundary_prompt_hard_memory_loss(...)` and adding a call-site regression fixed the training path; a two-step smoke then showed score-eligible/update-excluded fractions moving as expected. The invalid pre-fix op9 run should be ignored. The corrected op9 800-step preflight filled only score-eligible prompt memory (`55/55`) and reported route exclusion, but final exact/calc was only `0.510`, train prompt exact/calc `0.575`, heldout prompt exact/calc `0.050`, prior train/heldout `0.2909`/`0.0500`, and excluded route 1 overall calc `0.0385`. No trusted handoff was run because the source gate missed.

Questions:

- What did we learn about Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate?
- Has Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate been tested?
- Should we repeat Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate?
- What is the status of Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate?
- Why did Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-route-excluded-shared-prior-preflight.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not use the invalid pre-fix preflight as evidence, and do not run more short op9 route-exclusion preflights or route-heldout diagnostic ladders as novelty.

Next Allowed:

- Run the full op19 route-excluded shared-prior source using the strongest known numeric-prior dynamics and require source heldout/excluded-route quality before handoff, or replace it with a more explicitly shared/global prior objective.

Full Text:

```text
MIXED-NEGATIVE: Route-excluded prompt-memory training is wired, but the op9 shared-prior preflight misses the source gate.
Conclusion: A follow-up audit found the initial route-exclusion commit passed `memory_update_exclude_routes` to the fixed-batch online-memory branch, not the prompt-keyed branch used by the shared-prior source recipe. Moving it to `result_boundary_prompt_hard_memory_loss(...)` and adding a call-site regression fixed the training path; a two-step smoke then showed score-eligible/update-excluded fractions moving as expected. The invalid pre-fix op9 run should be ignored. The corrected op9 800-step preflight filled only score-eligible prompt memory (`55/55`) and reported route exclusion, but final exact/calc was only `0.510`, train prompt exact/calc `0.575`, heldout prompt exact/calc `0.050`, prior train/heldout `0.2909`/`0.0500`, and excluded route 1 overall calc `0.0385`. No trusted handoff was run because the source gate missed.
Do not repeat: Do not use the invalid pre-fix preflight as evidence, and do not run more short op9 route-exclusion preflights or route-heldout diagnostic ladders as novelty.
Next allowed test: Run the full op19 route-excluded shared-prior source using the strongest known numeric-prior dynamics and require source heldout/excluded-route quality before handoff, or replace it with a more explicitly shared/global prior objective.
Source: `aiAgentWorkHistory/phase7/2026-06-02-route-excluded-shared-prior-preflight.md`
```
