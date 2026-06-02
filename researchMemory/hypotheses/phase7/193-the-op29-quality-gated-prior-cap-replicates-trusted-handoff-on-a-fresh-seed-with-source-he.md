# The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap-fresh-seed.md

Summary:

- The same cap `2000` op29 h128 proportional recipe on CLI seed `31` / effective seed `33` froze through the quality gate at `2017` updates, `1,260,852` fit examples, and `1,080,000` full-fit examples. Source train exact/calc was `1.0000`, heldout exact/calc was lower than the original but still above the gate at `0.9111`, and heldout controls stayed low (`0.0333` injection-zero, `0.0000` forced-zero, `0.0167` forced-random). The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and final controls `0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random.

Questions:

- What did we learn about The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance?
- Has The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance been tested?
- Should we repeat The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance?
- What is the status of The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance?
- What follow-up is allowed for The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap-fresh-seed.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more cap-value or same-recipe seed ladders as novelty; this replicated handoff and exposed source-heldout variance.

Next Allowed:

- Many-calculator cost accounting for the capped recipe, or a less-prescriptive/non-enumerative credit mechanism that removes answer-derived candidate scoring.

Full Text:

```text
POSITIVE-WITH-CAVEAT: The op29 quality-gated prior cap replicates trusted handoff on a fresh seed, with source-heldout variance.
Conclusion: The same cap `2000` op29 h128 proportional recipe on CLI seed `31` / effective seed `33` froze through the quality gate at `2017` updates, `1,260,852` fit examples, and `1,080,000` full-fit examples. Source train exact/calc was `1.0000`, heldout exact/calc was lower than the original but still above the gate at `0.9111`, and heldout controls stayed low (`0.0333` injection-zero, `0.0000` forced-zero, `0.0167` forced-random). The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and final controls `0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random.
Do not repeat: Do not run more cap-value or same-recipe seed ladders as novelty; this replicated handoff and exposed source-heldout variance.
Next allowed test: Many-calculator cost accounting for the capped recipe, or a less-prescriptive/non-enumerative credit mechanism that removes answer-derived candidate scoring.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-quality-gated-prior-cap-fresh-seed.md`
```
