# Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates.

Kind: hypothesis_memory
Status: POSITIVE-WITH-CAVEAT
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-post-fill-full-refresh-prior-stress.md

Summary:

- Added `--result-boundary-target-amortized-prior-full-refresh-after-memory-full-updates`, which forces full-memory prior updates after prompt memory first fills before returning to the configured fit batch. On the op29 four-hook shared-output eval-only target-stratified h128 source, a `2500`-update full refresh reached overall exact/calc `0.9822`, train `0.9972`, heldout `0.9167`, prior train/heldout `0.9958`/`0.9167`, low heldout controls (`0.0278` injection-zero, `0.0000` forced-zero, `0.0111` forced-random), and `2755` total prior updates with `294,912` forced-result evals. The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000`, diagnostic calc `0.9922`, and low controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random).

Questions:

- What did we learn about Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates?
- Has Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates been tested?
- Should we repeat Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates?
- What is the status of Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates?
- What follow-up is allowed for Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-post-fill-full-refresh-prior-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same op29 full-refresh pass, constant batch160, h128-only capacity bumps, random fit-batch ladders, or validation threshold/patience ladders as novelty.

Next Allowed:

- Reduce or structure the full-refresh cost while preserving the same heldout source and trusted handoff gates: staged full refresh then coreset replay, coverage-aware/proportional fitting, refresh-stop/freeze transition, or explicit many-calculator cost accounting.

Full Text:

```text
POSITIVE-WITH-CAVEAT: Post-memory-fill full-refresh prior fitting clears the op29 source and trusted handoff gates.
Conclusion: Added `--result-boundary-target-amortized-prior-full-refresh-after-memory-full-updates`, which forces full-memory prior updates after prompt memory first fills before returning to the configured fit batch. On the op29 four-hook shared-output eval-only target-stratified h128 source, a `2500`-update full refresh reached overall exact/calc `0.9822`, train `0.9972`, heldout `0.9167`, prior train/heldout `0.9958`/`0.9167`, low heldout controls (`0.0278` injection-zero, `0.0000` forced-zero, `0.0111` forced-random), and `2755` total prior updates with `294,912` forced-result evals. The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000`, diagnostic calc `0.9922`, and low controls (`0.0000` injection-zero, `0.0000` forced-zero, `0.0078` forced-random).
Do not repeat: Do not rerun the same op29 full-refresh pass, constant batch160, h128-only capacity bumps, random fit-batch ladders, or validation threshold/patience ladders as novelty.
Next allowed test: Reduce or structure the full-refresh cost while preserving the same heldout source and trusted handoff gates: staged full refresh then coreset replay, coverage-aware/proportional fitting, refresh-stop/freeze transition, or explicit many-calculator cost accounting.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-post-fill-full-refresh-prior-stress.md`
```
