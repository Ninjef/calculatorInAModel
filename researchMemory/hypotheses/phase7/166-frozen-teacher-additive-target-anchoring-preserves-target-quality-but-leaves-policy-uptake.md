# Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-frozen-teacher-additive-target-anchor.md

Summary:

- Added `--result-boundary-target-teacher-checkpoint`, which constructs result-boundary targets with a separate frozen teacher while training the live model's result policy. Supporting probes showed the failure modes: freezing the whole encoder/readout preserved the repaired additive table (`best_true=0.5225`) but head-only policy uptake stalled (`learned_best=0.188`, final `0.0225`), while freezing only the post-calculator decoder let the pre-hook residual drift the target back down (`best_true=0.1575`, learned_best `0.6575`, final `0.0900`). The full frozen-teacher anchor preserved `best_true=0.5225` through 800 steps and improved learned-best to `0.4125`, but calculator-result accuracy/final eval reached only `0.1700`/`0.1750`, far below a useful source.

Questions:

- What did we learn about Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak?
- Has Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak been tested?
- Should we repeat Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak?
- What is the status of Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak?
- Why did Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-frozen-teacher-additive-target-anchor.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more same-checkpoint frozen-teacher additive-anchor length/LR/freezing sweeps as novelty. Target anchoring helps diagnose drift, but it does not solve policy uptake.

Next Allowed:

- Move to a different policy-uptake mechanism, such as a target that is easier for the policy class to represent, direct optimization of source logits against teacher target tables without additive forced-loss rescoring every step, or a new estimator that can raise true-result uptake while preserving the target table.

Full Text:

```text
MIXED-NEGATIVE: Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak.
Conclusion: Added `--result-boundary-target-teacher-checkpoint`, which constructs result-boundary targets with a separate frozen teacher while training the live model's result policy. Supporting probes showed the failure modes: freezing the whole encoder/readout preserved the repaired additive table (`best_true=0.5225`) but head-only policy uptake stalled (`learned_best=0.188`, final `0.0225`), while freezing only the post-calculator decoder let the pre-hook residual drift the target back down (`best_true=0.1575`, learned_best `0.6575`, final `0.0900`). The full frozen-teacher anchor preserved `best_true=0.5225` through 800 steps and improved learned-best to `0.4125`, but calculator-result accuracy/final eval reached only `0.1700`/`0.1750`, far below a useful source.
Do not repeat: Do not run more same-checkpoint frozen-teacher additive-anchor length/LR/freezing sweeps as novelty. Target anchoring helps diagnose drift, but it does not solve policy uptake.
Next allowed test: Move to a different policy-uptake mechanism, such as a target that is easier for the policy class to represent, direct optimization of source logits against teacher target tables without additive forced-loss rescoring every step, or a new estimator that can raise true-result uptake while preserving the target table.
Source: `aiAgentWorkHistory/phase7/2026-05-30-frozen-teacher-additive-target-anchor.md`
```
