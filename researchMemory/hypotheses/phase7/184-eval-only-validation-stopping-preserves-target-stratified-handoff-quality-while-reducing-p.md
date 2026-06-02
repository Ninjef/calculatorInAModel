# Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-01-target-stratified-eval-only-validation-stop-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-fit-validation-mode eval_only`, which keeps all prompt-memory entries in prior-fit updates and uses the validation split only for metrics/stopping. On a fresh effective seed13, target-stratified batch `160`, validation fraction `0.2`, stop metric `validation_accuracy`, threshold `0.9`, and patience `100` stopped at step `3250` with `1613` prior updates; source overall was `0.9825`, heldout `0.9500`, prior train/heldout `0.978125`/`0.9500`, and trusted handoff reached `1.0000` with low controls. On the same effective seed11 as the target-stratified benchmark, eval-only stopped at step `3600` with `1784` prior updates; source overall was `0.9725`, heldout `0.9125`, prior train/heldout `0.9625`/`0.9125`, and trusted handoff reached `1.0000` with low controls. Caveat: forced-result evals rose from the target-stratified seed11 benchmark `67,584` to `89,088` on seed11 and `124,416` on seed13 because prompt memory filled at step `100` instead of step `50`.

Questions:

- What did we learn about Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates?
- Has Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates been tested?
- Should we repeat Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates?
- What is the status of Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates?
- What follow-up is allowed for Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-eval-only-validation-stop-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Validation-heldout threshold/patience ladders; the meaningful distinction is eval-only validation versus excluding memory entries from fitting.

Next Allowed:

- Stress eval-only target-stratified stopping on a larger range and diagnose memory-fill/forced-eval behavior before promoting it to the default scalable recipe.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Eval-only validation stopping preserves target-stratified handoff quality while reducing prior updates.
Conclusion: Added `--result-boundary-target-amortized-prior-fit-validation-mode eval_only`, which keeps all prompt-memory entries in prior-fit updates and uses the validation split only for metrics/stopping. On a fresh effective seed13, target-stratified batch `160`, validation fraction `0.2`, stop metric `validation_accuracy`, threshold `0.9`, and patience `100` stopped at step `3250` with `1613` prior updates; source overall was `0.9825`, heldout `0.9500`, prior train/heldout `0.978125`/`0.9500`, and trusted handoff reached `1.0000` with low controls. On the same effective seed11 as the target-stratified benchmark, eval-only stopped at step `3600` with `1784` prior updates; source overall was `0.9725`, heldout `0.9125`, prior train/heldout `0.9625`/`0.9125`, and trusted handoff reached `1.0000` with low controls. Caveat: forced-result evals rose from the target-stratified seed11 benchmark `67,584` to `89,088` on seed11 and `124,416` on seed13 because prompt memory filled at step `100` instead of step `50`.
Do not repeat: Validation-heldout threshold/patience ladders; the meaningful distinction is eval-only validation versus excluding memory entries from fitting.
Next allowed test: Stress eval-only target-stratified stopping on a larger range and diagnose memory-fill/forced-eval behavior before promoting it to the default scalable recipe.
Source: `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-eval-only-validation-stop-gate.md`
```
