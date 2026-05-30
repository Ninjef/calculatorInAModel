# Static result-boundary set targets are paused after soft and regret-set negatives.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-result-boundary-set-target-steering-review.md

Summary:

- The target-construction escape hatch from the previous review has now been tested in its simplest static form. Soft-result targets and fixed-margin regret-set targets both weaken source acquisition relative to hard-best. Result-boundary remains a useful answer-derived bridge, but static broad target construction is now a local rut.

Questions:

- What did we learn about Static result-boundary set targets are paused after soft and regret-set negatives?
- Has Static result-boundary set targets are paused after soft and regret-set negatives been tested?
- Should we repeat Static result-boundary set targets are paused after soft and regret-set negatives?
- What is the status of Static result-boundary set targets are paused after soft and regret-set negatives?

Representative evidence:

- `researchReviews/2026-05-30-result-boundary-set-target-steering-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not tune static soft temperatures, fixed regret margins, or top-N low-regret static targets over the same full-loss table as mainline work.

Next Allowed:

- Evolving-checkpoint/streaming validation, calibrated proposal learning that preserves target quality while reducing scoring, adaptive uncertainty/regret selection, or a different credit-assignment family.

Full Text:

```text
REVIEW: Static result-boundary set targets are paused after soft and regret-set negatives.
Conclusion: The target-construction escape hatch from the previous review has now been tested in its simplest static form. Soft-result targets and fixed-margin regret-set targets both weaken source acquisition relative to hard-best. Result-boundary remains a useful answer-derived bridge, but static broad target construction is now a local rut.
Do not repeat: Do not tune static soft temperatures, fixed regret margins, or top-N low-regret static targets over the same full-loss table as mainline work.
Next allowed test: Evolving-checkpoint/streaming validation, calibrated proposal learning that preserves target quality while reducing scoring, adaptive uncertainty/regret selection, or a different credit-assignment family.
Source: `researchReviews/2026-05-30-result-boundary-set-target-steering-review.md`
```
