# Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-eval-only-target-stratified-prior-stress.md

Summary:

- Stressed the op19 eval-only target-stratified recipe at `operand_max=29` with four routed shared-output hooks, h64 numeric prior, fit batch `160`, eval-only validation, and 5000 streaming source steps. Prompt memory filled at step `200` after `290,304` forced-result evals and the source reached high train accuracy (`0.9931`), but heldout exact/calc was only `0.8444` and the prior itself reached only train/heldout `0.8375`/`0.7667`; the validation stop never fired and prior updates stayed at `2501`. Post-hoc full-memory diagnostics from the same trace showed the discovered targets were mostly good (`0.9931` train targets matched true sums) and that longer/richer prior fitting can recover much of the gap: h64 full-memory fit for `2500` steps reached train/heldout `0.9875`/`0.9000`, and h128 reached `0.9889`/`0.9278`. The op29 miss is therefore a prior-fitting/capacity scaling problem more than a memory-fill or calculator-wiring problem.

Questions:

- What did we learn about Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29?
- Has Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29 been tested?
- Should we repeat Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29?
- What is the status of Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29?
- Why did Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29 fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-eval-only-target-stratified-prior-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run another op29 constant `fit_batch_size=160` eval-only repeat, random fit-batch ladder, validation-heldout threshold/patience ladder, or trusted handoff from this heldout-missed source as progress.

Next Allowed:

- Change the prior mechanism or fit dynamics: e.g. richer numeric features/capacity, post-memory-fill full refresh, or coverage-aware/proportional fitting with explicit cost accounting before another full source/handoff gate.

Full Text:

```text
MIXED-NEGATIVE: Constant-size target-stratified eval-only prior fitting does not scale cleanly to op29.
Conclusion: Stressed the op19 eval-only target-stratified recipe at `operand_max=29` with four routed shared-output hooks, h64 numeric prior, fit batch `160`, eval-only validation, and 5000 streaming source steps. Prompt memory filled at step `200` after `290,304` forced-result evals and the source reached high train accuracy (`0.9931`), but heldout exact/calc was only `0.8444` and the prior itself reached only train/heldout `0.8375`/`0.7667`; the validation stop never fired and prior updates stayed at `2501`. Post-hoc full-memory diagnostics from the same trace showed the discovered targets were mostly good (`0.9931` train targets matched true sums) and that longer/richer prior fitting can recover much of the gap: h64 full-memory fit for `2500` steps reached train/heldout `0.9875`/`0.9000`, and h128 reached `0.9889`/`0.9278`. The op29 miss is therefore a prior-fitting/capacity scaling problem more than a memory-fill or calculator-wiring problem.
Do not repeat: Do not run another op29 constant `fit_batch_size=160` eval-only repeat, random fit-batch ladder, validation-heldout threshold/patience ladder, or trusted handoff from this heldout-missed source as progress.
Next allowed test: Change the prior mechanism or fit dynamics: e.g. richer numeric features/capacity, post-memory-fill full refresh, or coverage-aware/proportional fitting with explicit cost accounting before another full source/handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-eval-only-target-stratified-prior-stress.md`
```
