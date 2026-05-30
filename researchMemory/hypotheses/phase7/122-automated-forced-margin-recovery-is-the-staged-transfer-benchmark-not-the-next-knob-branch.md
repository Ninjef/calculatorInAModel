# Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch.

Kind: hypothesis_memory
Status: REVIEW
Phase: Phase 7
Source: researchReviews/2026-05-30-forced-margin-benchmark-direction-review.md

Summary:

- The post-recovery forced-margin branch now has enough evidence to stop local expansion. Manual recovery reached `0.8700` final / `0.9050` step-600 handoff; first automated fresh seed reached `0.9875` / `0.9800`; second fresh seed reached `0.8975` / `0.9050`, exposing variance; and a wider `n_embd=32`, `n_head=2` non-product decoder stress reached `1.0000` final / `1.0000` step-600 handoff with low controls. This makes automated one-negative forced-margin recovery the benchmark to beat for staged transfer, but it still depends on hard improvement assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.

Questions:

- What did we learn about Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch?
- Has Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch been tested?
- Should we repeat Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch?
- What is the status of Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch?
- Why did Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch fail?

Representative evidence:

- `researchReviews/2026-05-30-forced-margin-benchmark-direction-review.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more local forced-margin start-step, margin, negative-count, LR, recovery-length, same-scale seed-only, cheap-selector, or same wider non-product stress variants as novelty.

Next Allowed:

- Forced-margin compute should stress a new thesis-relevant axis such as product-decoder parity, larger operand range, larger architecture, or many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.

Full Text:

```text
REVIEW: Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch.
Conclusion: The post-recovery forced-margin branch now has enough evidence to stop local expansion. Manual recovery reached `0.8700` final / `0.9050` step-600 handoff; first automated fresh seed reached `0.9875` / `0.9800`; second fresh seed reached `0.8975` / `0.9050`, exposing variance; and a wider `n_embd=32`, `n_head=2` non-product decoder stress reached `1.0000` final / `1.0000` step-600 handoff with low controls. This makes automated one-negative forced-margin recovery the benchmark to beat for staged transfer, but it still depends on hard improvement assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.
Do not repeat: Do not run more local forced-margin start-step, margin, negative-count, LR, recovery-length, same-scale seed-only, cheap-selector, or same wider non-product stress variants as novelty.
Next allowed test: Forced-margin compute should stress a new thesis-relevant axis such as product-decoder parity, larger operand range, larger architecture, or many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.
Source: `researchReviews/2026-05-30-forced-margin-benchmark-direction-review.md`
```
