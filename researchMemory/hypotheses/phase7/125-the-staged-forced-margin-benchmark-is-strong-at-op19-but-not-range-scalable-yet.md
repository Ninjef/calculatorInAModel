# The staged forced-margin benchmark is strong at op19 but not range-scalable yet.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-forced-margin-range-stress-review.md

Summary:

- Product-decoder parity removed the wider-decoder caveat at `operand_max=19`, but the first larger-range stress at op29 did not clear. The op29 oracle product decoder reached `1.0000`, while source acquisition plateaued far lower and the trusted handoff reached only `0.8533` final despite low ablation controls. This makes range scaling an unresolved source-acquisition/assignment-cost problem, not a decoder/readout wiring problem.

Questions:

- What did we learn about The staged forced-margin benchmark is strong at op19 but not range-scalable yet?
- Has The staged forced-margin benchmark is strong at op19 but not range-scalable yet been tested?
- Should we repeat The staged forced-margin benchmark is strong at op19 but not range-scalable yet?
- What is the status of The staged forced-margin benchmark is strong at op19 but not range-scalable yet?

Representative evidence:

- `researchReviews/2026-05-30-forced-margin-range-stress-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not respond to the op29 miss with local forced-margin knob tuning, same-seed reruns, or op49 full-grid repetition as novelty.

Next Allowed:

- Prioritize changed source objectives, scalable assignment approximations against the exact-grid ceiling, or a diagnostic source-capacity/recovery test that explains the op29 failure before larger-range runs.

Full Text:

```text
REVIEW: The staged forced-margin benchmark is strong at op19 but not range-scalable yet.
Conclusion: Product-decoder parity removed the wider-decoder caveat at `operand_max=19`, but the first larger-range stress at op29 did not clear. The op29 oracle product decoder reached `1.0000`, while source acquisition plateaued far lower and the trusted handoff reached only `0.8533` final despite low ablation controls. This makes range scaling an unresolved source-acquisition/assignment-cost problem, not a decoder/readout wiring problem.
Do not repeat: Do not respond to the op29 miss with local forced-margin knob tuning, same-seed reruns, or op49 full-grid repetition as novelty.
Next allowed test: Prioritize changed source objectives, scalable assignment approximations against the exact-grid ceiling, or a diagnostic source-capacity/recovery test that explains the op29 failure before larger-range runs.
Source: `researchReviews/2026-05-30-forced-margin-range-stress-review.md`
```
