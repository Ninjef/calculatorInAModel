# Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-error-stratified-coreset-refresh-gate.md

Summary:

- Added `error_stratified` amortized-prior fit sampling, which prioritizes currently misclassified prompt-memory entries and then fills the batch target-stratified. With a shorter `1500` full-refresh window, error-stratified batch `160`, and the dual train+validation stop guard, the op29 source still reached overall exact/calc `0.9922`, train `1.0000`, heldout `0.9556`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and low controls (`0.0078` injection-zero, `0.0000` forced-zero, `0.0391` forced-random). But the prior never met the stop gate, final prior train/heldout were only `0.8806`/`0.8778`, and prior updates rose to `3251` with `302,592` forced-result evals, worse than both full-refresh positives.

Questions:

- What did we learn about Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost?
- Has Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost been tested?
- Should we repeat Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost?
- What is the status of Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost?
- Why did Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost fail?
- What follow-up is allowed for Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-error-stratified-coreset-refresh-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run error-focused coreset batch-size, refresh-window, or threshold ladders as novelty.

Next Allowed:

- A different cost structure, not harder sampling of current errors: e.g. coverage-aware/proportional refresh with explicit update caps, staged refresh plus stable coreset distillation, or many-calculator cost accounting.

Full Text:

```text
MIXED-NEGATIVE: Error-stratified coreset replay preserves op29 source/handoff but worsens prior-update cost.
Conclusion: Added `error_stratified` amortized-prior fit sampling, which prioritizes currently misclassified prompt-memory entries and then fills the batch target-stratified. With a shorter `1500` full-refresh window, error-stratified batch `160`, and the dual train+validation stop guard, the op29 source still reached overall exact/calc `0.9922`, train `1.0000`, heldout `0.9556`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and low controls (`0.0078` injection-zero, `0.0000` forced-zero, `0.0391` forced-random). But the prior never met the stop gate, final prior train/heldout were only `0.8806`/`0.8778`, and prior updates rose to `3251` with `302,592` forced-result evals, worse than both full-refresh positives.
Do not repeat: Do not run error-focused coreset batch-size, refresh-window, or threshold ladders as novelty.
Next allowed test: A different cost structure, not harder sampling of current errors: e.g. coverage-aware/proportional refresh with explicit update caps, staged refresh plus stable coreset distillation, or many-calculator cost accounting.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-error-stratified-coreset-refresh-gate.md`
```
