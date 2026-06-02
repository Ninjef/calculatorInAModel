# Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-dual-stop-refresh-cost-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-stop-require-train-accuracy`, so validation stop can require train-memory prior coverage before ending a full-refresh window. On a fresh/effective op29 h128 source with validation `>=0.9`, train requirement `>=0.98`, and patience `100`, the source reached overall exact/calc `0.9956`, train `1.0000`, heldout `0.9667`, prior train/heldout `0.9972`/`0.9667`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). The rule stopped at `2570` prior updates after `278,016` forced-result evals, versus `2755` updates in the prior full-refresh positive. Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and low controls (`0.0000` injection-zero, `0.0078` forced-zero, `0.0234` forced-random).

Questions:

- What did we learn about Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut?
- Has Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut been tested?
- Should we repeat Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut?
- What is the status of Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut?
- What follow-up is allowed for Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-dual-stop-refresh-cost-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run train-requirement threshold/patience ladders or same-recipe seed repeats as novelty; the gain is only `185` prior updates.

Next Allowed:

- A higher-leverage structured refresh-cost mechanism: staged full refresh then coreset replay, coverage-aware/proportional refresh, or many-calculator cost accounting that targets a larger update/forced-eval reduction while keeping the source/handoff gates.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Dual train+validation stopping during op29 full refresh preserves source and handoff with a small update cut.
Conclusion: Added `--result-boundary-target-amortized-prior-stop-require-train-accuracy`, so validation stop can require train-memory prior coverage before ending a full-refresh window. On a fresh/effective op29 h128 source with validation `>=0.9`, train requirement `>=0.98`, and patience `100`, the source reached overall exact/calc `0.9956`, train `1.0000`, heldout `0.9667`, prior train/heldout `0.9972`/`0.9667`, and low heldout controls (`0.0222` injection-zero, `0.0000` forced-zero, `0.0056` forced-random). The rule stopped at `2570` prior updates after `278,016` forced-result evals, versus `2755` updates in the prior full-refresh positive. Trusted 600-step additive handoff reached `900/900 = 1.0000`, diagnostic exact/calc `1.0000`/`0.9844`, and low controls (`0.0000` injection-zero, `0.0078` forced-zero, `0.0234` forced-random).
Do not repeat: Do not run train-requirement threshold/patience ladders or same-recipe seed repeats as novelty; the gain is only `185` prior updates.
Next allowed test: A higher-leverage structured refresh-cost mechanism: staged full refresh then coreset replay, coverage-aware/proportional refresh, or many-calculator cost accounting that targets a larger update/forced-eval reduction while keeping the source/handoff gates.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-dual-stop-refresh-cost-gate.md`
```
