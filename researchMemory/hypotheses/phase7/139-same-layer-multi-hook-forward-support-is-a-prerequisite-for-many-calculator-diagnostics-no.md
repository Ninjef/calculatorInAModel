# Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-same-layer-multi-hook-forward-support.md

Summary:

- Added `GPTConfig.calculator_hook_count`, independent extra calculator hooks, combined same-layer injections, diagnostics for `calculator_active_hook_count` and per-hook injections, `--calculator-hook-count` in `scripts/overfit_one_batch.py`, and optimizer/freezing support so extra hook policy heads are not silently treated as upstream. Tests verify injection summation and multi-hook freezing/grouping. A zero-step smoke with `--calculator-hook-count 3` wrote `calculator_hook_count=3` in config and metrics and grouped hook input projections separately from upstream.

Questions:

- What did we learn about Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training?
- Has Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training been tested?
- Should we repeat Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training?
- What is the status of Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training?
- What follow-up is allowed for Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-same-layer-multi-hook-forward-support.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not claim many-calculator success from this alone. It does not provide routed/scattered hook placement, per-hook task specialization, or evidence that independent calculator policies train under assignment pressure.

Next Allowed:

- Build a routed or task-partitioned multi-hook diagnostic that measures active hooks, scorer calls, per-hook policy quality, and leakage/interference; alternatively move to a non-enumerative credit signal that makes hook count less central.

Full Text:

```text
PARTIAL: Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training.
Conclusion: Added `GPTConfig.calculator_hook_count`, independent extra calculator hooks, combined same-layer injections, diagnostics for `calculator_active_hook_count` and per-hook injections, `--calculator-hook-count` in `scripts/overfit_one_batch.py`, and optimizer/freezing support so extra hook policy heads are not silently treated as upstream. Tests verify injection summation and multi-hook freezing/grouping. A zero-step smoke with `--calculator-hook-count 3` wrote `calculator_hook_count=3` in config and metrics and grouped hook input projections separately from upstream.
Do not repeat: Do not claim many-calculator success from this alone. It does not provide routed/scattered hook placement, per-hook task specialization, or evidence that independent calculator policies train under assignment pressure.
Next allowed test: Build a routed or task-partitioned multi-hook diagnostic that measures active hooks, scorer calls, per-hook policy quality, and leakage/interference; alternatively move to a non-enumerative credit signal that makes hook count less central.
Source: `aiAgentWorkHistory/phase7/2026-05-30-same-layer-multi-hook-forward-support.md`
```
