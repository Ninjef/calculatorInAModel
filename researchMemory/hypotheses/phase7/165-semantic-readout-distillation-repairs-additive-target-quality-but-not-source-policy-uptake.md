# Semantic readout distillation repairs additive target quality but not source-policy uptake.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-semantic-distilled-additive-zero-improvement.md

Summary:

- Added `--additive-semantic-distill-*`, which forces arbitrary result classes and trains the additive non-bottleneck path to match the frozen answer-decoder bottleneck logits. Co-training with additive zero-improvement improved the target slightly (`hard_best_equals_true_sum 0.0325 -> 0.1775`) but final calc stayed weak (`0.0825` snapshot / `0.0450` eval). A 300-step distill-only preconditioner raised teacher/student token agreement to `0.7694`; starting source training from that checkpoint repaired additive target quality (`best=true 0.5225` at step 0 and `0.8200` at step 200 with ongoing distill), but learned-best/source calc stayed low (`0.1400`/`0.0675`). Turning distill off let the policy learn the now-drifting non-arithmetic target (`learned_best=0.6950`, best=true fell to `0.1575`, calc `0.0900`). Distillation teaches readout semantics, but source-policy uptake still needs a stronger mechanism.

Questions:

- What did we learn about Semantic readout distillation repairs additive target quality but not source-policy uptake?
- Has Semantic readout distillation repairs additive target quality but not source-policy uptake been tested?
- Should we repeat Semantic readout distillation repairs additive target quality but not source-policy uptake?
- What is the status of Semantic readout distillation repairs additive target quality but not source-policy uptake?
- Why did Semantic readout distillation repairs additive target quality but not source-policy uptake fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-semantic-distilled-additive-zero-improvement.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more plain semantic-distill weight/sample-count/length tweaks as novelty; the next variant must address policy uptake or target drift explicitly.

Next Allowed:

- Couple readout distillation to a policy-learning mechanism such as staged frozen-readout target construction, policy-target anchoring to the repaired table, or an estimator that preserves target quality while increasing learned-best/true-result uptake.

Full Text:

```text
MIXED-NEGATIVE: Semantic readout distillation repairs additive target quality but not source-policy uptake.
Conclusion: Added `--additive-semantic-distill-*`, which forces arbitrary result classes and trains the additive non-bottleneck path to match the frozen answer-decoder bottleneck logits. Co-training with additive zero-improvement improved the target slightly (`hard_best_equals_true_sum 0.0325 -> 0.1775`) but final calc stayed weak (`0.0825` snapshot / `0.0450` eval). A 300-step distill-only preconditioner raised teacher/student token agreement to `0.7694`; starting source training from that checkpoint repaired additive target quality (`best=true 0.5225` at step 0 and `0.8200` at step 200 with ongoing distill), but learned-best/source calc stayed low (`0.1400`/`0.0675`). Turning distill off let the policy learn the now-drifting non-arithmetic target (`learned_best=0.6950`, best=true fell to `0.1575`, calc `0.0900`). Distillation teaches readout semantics, but source-policy uptake still needs a stronger mechanism.
Do not repeat: Do not run more plain semantic-distill weight/sample-count/length tweaks as novelty; the next variant must address policy uptake or target drift explicitly.
Next allowed test: Couple readout distillation to a policy-learning mechanism such as staged frozen-readout target construction, policy-target anchoring to the repaired table, or an estimator that preserves target quality while increasing learned-best/true-result uptake.
Source: `aiAgentWorkHistory/phase7/2026-05-30-semantic-distilled-additive-zero-improvement.md`
```
