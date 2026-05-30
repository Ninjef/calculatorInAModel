# Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md

Summary:

- A new diagnostic trained a shared MLP critic on sparse forced-result scores using prompt hidden-state features plus candidate calculator output vectors, then evaluated whether predicted losses recover the full-enum result-boundary argmin on heldout prompts. The full-enum boundary target was valid at all checked checkpoints (`1.0000` best=true-sum), but sparse critic recovery was poor: with `8` scores per train prompt, heldout argmin recovery was `0.0800` at step `0`, `0.0800` at step `100`, and `0.1700` at step `800`; with `24` scores per train prompt, it was still only `0.2600` at step `0` and `0.1900` at step `800`. Do not treat this pointwise hidden/output critic as a scalable replacement for full result-boundary enumeration.

Questions:

- What did we learn about Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets?
- Has Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets been tested?
- Should we repeat Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets?
- What is the status of Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets?
- Why did Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same hidden-state plus candidate-output-vector pointwise MLP critic on the May 13 result-boundary source checkpoints with `k=8` or `k=24` sparse scores per prompt as novelty.

Next Allowed:

- Continue result-boundary approximation only with a stronger mechanism: rank/contrastive or uncertainty-aware critic objectives, feature validation tied to evolving model states, or a different target construction that does not require pointwise loss prediction to identify the exact argmin.

Full Text:

```text
MIXED-NEGATIVE: Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets.
Conclusion: A new diagnostic trained a shared MLP critic on sparse forced-result scores using prompt hidden-state features plus candidate calculator output vectors, then evaluated whether predicted losses recover the full-enum result-boundary argmin on heldout prompts. The full-enum boundary target was valid at all checked checkpoints (`1.0000` best=true-sum), but sparse critic recovery was poor: with `8` scores per train prompt, heldout argmin recovery was `0.0800` at step `0`, `0.0800` at step `100`, and `0.1700` at step `800`; with `24` scores per train prompt, it was still only `0.2600` at step `0` and `0.1900` at step `800`. Do not treat this pointwise hidden/output critic as a scalable replacement for full result-boundary enumeration.
Do not repeat: Do not rerun the same hidden-state plus candidate-output-vector pointwise MLP critic on the May 13 result-boundary source checkpoints with `k=8` or `k=24` sparse scores per prompt as novelty.
Next allowed test: Continue result-boundary approximation only with a stronger mechanism: rank/contrastive or uncertainty-aware critic objectives, feature validation tied to evolving model states, or a different target construction that does not require pointwise loss prediction to identify the exact argmin.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md`
```
