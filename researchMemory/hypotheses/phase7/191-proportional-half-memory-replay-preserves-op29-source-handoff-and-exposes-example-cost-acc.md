# Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-proportional-half-memory-refresh-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-fit-batch-fraction` and cumulative prior fit-example metrics, then ran op29 h128 with a shorter `1500` full-memory refresh followed by target-stratified half-memory fits (`0.5`, effective batch `360`) under the dual train+validation guard. The source reached overall exact/calc `0.9933`, train `1.0000`, heldout `0.9556`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9922`, and low final snapshot controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0156` forced-random). Cost accounting reported `3251` prior updates, `1,705,177` fit examples, `1,080,000` full-fit examples, and no stop; final prior train/validation were `0.9583`/`0.9635`.

Questions:

- What did we learn about Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence?
- Has Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence been tested?
- Should we repeat Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence?
- What is the status of Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence?
- What follow-up is allowed for Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-proportional-half-memory-refresh-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run proportional fraction or refresh-window ladders as novelty. The pass is real, but the update count is still high and the example-cost reduction is modest.

Next Allowed:

- Add an explicit update cap/freeze after a validated proportional replay phase, distill the stable coreset into the prior, or shift to many-calculator cost accounting/new credit assignment.

Full Text:

```text
MIXED-POSITIVE: Proportional half-memory replay preserves op29 source/handoff and exposes example-cost accounting, but not update convergence.
Conclusion: Added `--result-boundary-target-amortized-prior-fit-batch-fraction` and cumulative prior fit-example metrics, then ran op29 h128 with a shorter `1500` full-memory refresh followed by target-stratified half-memory fits (`0.5`, effective batch `360`) under the dual train+validation guard. The source reached overall exact/calc `0.9933`, train `1.0000`, heldout `0.9556`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9922`, and low final snapshot controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0156` forced-random). Cost accounting reported `3251` prior updates, `1,705,177` fit examples, `1,080,000` full-fit examples, and no stop; final prior train/validation were `0.9583`/`0.9635`.
Do not repeat: Do not run proportional fraction or refresh-window ladders as novelty. The pass is real, but the update count is still high and the example-cost reduction is modest.
Next allowed test: Add an explicit update cap/freeze after a validated proportional replay phase, distill the stable coreset into the prior, or shift to many-calculator cost accounting/new credit assignment.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-proportional-half-memory-refresh-gate.md`
```
