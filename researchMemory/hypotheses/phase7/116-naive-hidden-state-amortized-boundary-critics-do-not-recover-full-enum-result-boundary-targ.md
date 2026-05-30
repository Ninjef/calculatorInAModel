# Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets.

Status: MIXED-NEGATIVE.

Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md

Summary:

- Added `scripts/diagnose_result_boundary_amortized_critic.py` to test whether
  a shared critic trained from sparse forced-result scores can approximate the
  full result-boundary target on heldout prompts.
- The critic uses prompt hidden-state features plus candidate calculator output
  vectors, avoiding explicit operand-polynomial features.
- On the known result-boundary source lineage, the full-enum best result was
  the true sum for all heldout prompts at step `0`, step `100`, and step `800`.
- With `8` forced scores per train prompt, heldout argmin recovery was only
  `0.0800`, `0.0800`, and `0.1700` at steps `0`, `100`, and `800`.
- With `24` forced scores per train prompt, heldout argmin recovery was still
  only `0.2600` at step `0` and `0.1900` at step `800`.
- This makes the naive pointwise hidden/output critic a poor candidate for
  replacing full forced-result enumeration in result-boundary target training.

Questions this memory answers:

- Can a simple hidden-state amortized critic replace full result-boundary
  enumeration?
- Did sparse hidden/output loss prediction recover heldout boundary argmins?
- Should future agents wire this exact critic into source training?
- What sparse budgets were checked for result-boundary critic approximation?

Do not repeat:

- Do not rerun the same hidden-state plus candidate-output-vector pointwise MLP
  critic on the May 13 result-boundary source checkpoints with `k=8` or `k=24`
  sparse scores per prompt as novelty.

Next allowed test:

- Continue result-boundary approximation only with a stronger mechanism:
  rank/contrastive or uncertainty-aware critic objectives, feature validation
  tied to evolving model states, or a different target construction that does
  not require pointwise loss prediction to identify the exact argmin.

Ledger entry:

MIXED-NEGATIVE: Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets. Conclusion: A new diagnostic trained a shared MLP critic on sparse forced-result scores using prompt hidden-state features plus candidate calculator output vectors, then evaluated whether predicted losses recover the full-enum result-boundary argmin on heldout prompts. The full-enum boundary target was valid at all checked checkpoints (`1.0000` best=true-sum), but sparse critic recovery was poor: with `8` scores per train prompt, heldout argmin recovery was `0.0800` at step `0`, `0.0800` at step `100`, and `0.1700` at step `800`; with `24` scores per train prompt, it was still only `0.2600` at step `0` and `0.1900` at step `800`. Do not treat this pointwise hidden/output critic as a scalable replacement for full result-boundary enumeration.
Do not repeat: Do not rerun the same hidden-state plus candidate-output-vector pointwise MLP critic on the May 13 result-boundary source checkpoints with `k=8` or `k=24` sparse scores per prompt as novelty.
Next allowed test: Continue result-boundary approximation only with a stronger mechanism: rank/contrastive or uncertainty-aware critic objectives, feature validation tied to evolving model states, or a different target construction that does not require pointwise loss prediction to identify the exact argmin.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md`
