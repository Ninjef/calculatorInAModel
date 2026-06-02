# Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-01-target-stratified-validation-stop-gate.md

Summary:

- Added validation-split prior fitting and a validation-accuracy stop metric. With target-stratified batch `160`, validation fraction `0.2`, stop metric `validation_accuracy`, threshold `0.9`, and patience `100`, the source stopped at `2359` prior updates and `53,760` forced evals, but heldout exact/calc fell to `0.8625` and overall to `0.9725`. Train remained high at `0.990625`, and controls stayed low, so this is a prior/generalization quality miss rather than calculator wiring failure. No trusted handoff was run because the source gate missed.

Questions:

- What did we learn about Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate?
- Has Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate been tested?
- Should we repeat Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate?
- What is the status of Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate?
- Why did Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-validation-stop-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Validation-heldout prompt-memory threshold/patience ladders as novelty; removing 20% of memory entries from prior fitting weakened the source.

Next Allowed:

- If using a validation signal, use eval-only validation without excluding entries from fitting, or use a rolling/full-fit target-stratified stop. Otherwise stress the positive target-stratified coreset on a fresh seed/range axis.

Full Text:

```text
MIXED-NEGATIVE: Validation-heldout target-stratified prior stopping reduces updates but misses the heldout source gate.
Conclusion: Added validation-split prior fitting and a validation-accuracy stop metric. With target-stratified batch `160`, validation fraction `0.2`, stop metric `validation_accuracy`, threshold `0.9`, and patience `100`, the source stopped at `2359` prior updates and `53,760` forced evals, but heldout exact/calc fell to `0.8625` and overall to `0.9725`. Train remained high at `0.990625`, and controls stayed low, so this is a prior/generalization quality miss rather than calculator wiring failure. No trusted handoff was run because the source gate missed.
Do not repeat: Validation-heldout prompt-memory threshold/patience ladders as novelty; removing 20% of memory entries from prior fitting weakened the source.
Next allowed test: If using a validation signal, use eval-only validation without excluding entries from fitting, or use a rolling/full-fit target-stratified stop. Otherwise stress the positive target-stratified coreset on a fresh seed/range axis.
Source: `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-validation-stop-gate.md`
```
