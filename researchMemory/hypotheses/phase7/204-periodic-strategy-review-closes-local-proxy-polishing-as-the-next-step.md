# Periodic strategy review closes local proxy polishing as the next step.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-06-03-periodic-strategy-review.md

Summary:

- Reviewed the recent family-14, many-calculator accounting, route-heldout, route-excluded, prior-bootstrap, candidate-evidence, and background-refresh evidence. The pattern is repeated: local mechanisms can make current bottleneck sources and handoffs work, but the remaining failures are not fixed by more seed/threshold/cost ladders. The current recipe still depends on per-calculator prompt-memory target tables, answer-derived candidate scoring, and staged frozen-policy transfer; capped priors improve cost but still scale linearly with independent calculators, and route-excluded variants show the live source process cannot yet share targets well enough when direct route discovery is removed.

Questions:

- What did we learn about Periodic strategy review closes local proxy polishing as the next step?
- Has Periodic strategy review closes local proxy polishing as the next step been tested?
- Should we repeat Periodic strategy review closes local proxy polishing as the next step?
- What is the status of Periodic strategy review closes local proxy polishing as the next step?

Representative evidence:

- `researchReviews/2026-06-03-periodic-strategy-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not count cap/fraction/window/seed ladders, route-excluded patch variants, source-checkpoint selectors, or static local-target/proposal variants as new algorithmic progress.

Next Allowed:

- Require any new run to change target formation or credit assignment itself: a shared/global target model, joint cross-route target learning, amortization across calculators that removes per-calculator prompt tables, or a less-prescriptive credit signal that bypasses answer-derived candidate scoring.

Full Text:

```text
REVIEW: Periodic strategy review closes local proxy polishing as the next step.
Conclusion: Reviewed the recent family-14, many-calculator accounting, route-heldout, route-excluded, prior-bootstrap, candidate-evidence, and background-refresh evidence. The pattern is repeated: local mechanisms can make current bottleneck sources and handoffs work, but the remaining failures are not fixed by more seed/threshold/cost ladders. The current recipe still depends on per-calculator prompt-memory target tables, answer-derived candidate scoring, and staged frozen-policy transfer; capped priors improve cost but still scale linearly with independent calculators, and route-excluded variants show the live source process cannot yet share targets well enough when direct route discovery is removed.
Do not repeat: Do not count cap/fraction/window/seed ladders, route-excluded patch variants, source-checkpoint selectors, or static local-target/proposal variants as new algorithmic progress.
Next allowed test: Require any new run to change target formation or credit assignment itself: a shared/global target model, joint cross-route target learning, amortization across calculators that removes per-calculator prompt tables, or a less-prescriptive credit signal that bypasses answer-derived candidate scoring.
Source: `researchReviews/2026-06-03-periodic-strategy-review.md`
```
