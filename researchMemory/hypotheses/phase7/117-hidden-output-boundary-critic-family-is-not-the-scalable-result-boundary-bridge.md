# Hidden-output boundary critic family is not the scalable result-boundary bridge.

Status: REVIEW, paused.

Source: researchReviews/2026-05-30-result-boundary-approximation-review.md

Summary:

- The result-boundary source transfer positive makes answer-derived targets
  strategically interesting, but simple amortized critics did not provide a
  scalable replacement for full forced-result enumeration.
- Pointwise hidden/output critic recovery was poor: `k=8` heldout argmin
  recovery was `0.0800`, `0.0800`, `0.1700` at steps `0`, `100`, `800`; `k=24`
  was `0.2600` at step `0` and `0.1900` at step `800`.
- Pairwise ranking helped at the trained checkpoint but was still insufficient:
  pairwise `k=24` reached `0.4000` heldout argmin recovery at step `800` and
  `0.2600` at step `0`; hybrid was worse (`0.2700`/`0.2000`).
- Since `k=24` is already most of the `39`-class result vocabulary, this critic
  family does not look like a viable scalable bridge.

Questions this memory answers:

- Should future agents tune hidden-output result-boundary critic losses?
- Did pairwise/rank-aware critic training solve the amortized boundary target?
- Is top-5 recovery enough to wire the critic into source training?
- What result-boundary approximation branch is paused?

Do not repeat:

- Do not continue pointwise, pairwise, hybrid, hidden-size, epoch-count, or
  learning-rate variants of the same hidden-state plus candidate-output-vector
  critic as novelty.

Next allowed test:

- Continue answer-derived result-boundary work only with a different target
  construction, uncertainty-aware compute allocation, or a generalization
  mechanism validated across evolving model states or prompt ranges.

Ledger entry:

REVIEW: Hidden-output boundary critic family is not the scalable result-boundary bridge. Conclusion: The answer-derived result-boundary source transfer remains strategically useful, but the simple amortized critic family should pause. Pointwise hidden/output critics recovered heldout full-enum argmins only `0.0800-0.1700` at `k=8` and `0.1900-0.2600` at `k=24`. Pairwise ranking improved the trained step-800 checkpoint to `0.4000` argmin recovery at `k=24`, but step `0` stayed `0.2600`, hybrid was worse, and `k=24` already uses most of the 39-class result vocabulary. This is not a practical replacement for full enumeration.
Do not repeat: Do not continue pointwise, pairwise, hybrid, hidden-size, epoch-count, or learning-rate variants of the same hidden-state plus candidate-output-vector critic as novelty.
Next allowed test: Continue answer-derived result-boundary work only with a different target construction, uncertainty-aware compute allocation, or a generalization mechanism validated across evolving model states or prompt ranges.
Source: `researchReviews/2026-05-30-result-boundary-approximation-review.md`
