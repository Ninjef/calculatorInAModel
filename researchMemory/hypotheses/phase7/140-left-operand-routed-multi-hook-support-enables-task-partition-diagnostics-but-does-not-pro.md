# Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-left-operand-routed-multi-hook-support.md

Summary:

- Added `calculator_hook_routing='left_operand_mod'`, which routes each fixed-width prompt to one active same-layer hook by final left-operand digit modulo `calculator_hook_count`. Diagnostics now report `calculator_hook_route` and `calculator_hook_route_counts`, and per-hook applied injections are zeroed for non-routed examples. `scripts/overfit_one_batch.py` exposes `--calculator-hook-routing left_operand_mod`; a zero-step smoke with `--calculator-hook-count 3 --calculator-hook-routing left_operand_mod` wrote matching routing/count fields in `config.json` and `metrics.json`.

Questions:

- What did we learn about Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training?
- Has Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training been tested?
- Should we repeat Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training?
- What is the status of Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training?
- What follow-up is allowed for Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-left-operand-routed-multi-hook-support.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat routing support or zero-step smoke as evidence that independent hooks can learn specialized calculator policies. It only makes a routed diagnostic possible.

Next Allowed:

- Run a small task-partitioned training diagnostic that reports per-hook route counts, per-hook calculator-result accuracy, scorer calls under topk/exact assignment, and whether routed hooks interfere or specialize.

Full Text:

```text
PARTIAL: Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training.
Conclusion: Added `calculator_hook_routing='left_operand_mod'`, which routes each fixed-width prompt to one active same-layer hook by final left-operand digit modulo `calculator_hook_count`. Diagnostics now report `calculator_hook_route` and `calculator_hook_route_counts`, and per-hook applied injections are zeroed for non-routed examples. `scripts/overfit_one_batch.py` exposes `--calculator-hook-routing left_operand_mod`; a zero-step smoke with `--calculator-hook-count 3 --calculator-hook-routing left_operand_mod` wrote matching routing/count fields in `config.json` and `metrics.json`.
Do not repeat: Do not treat routing support or zero-step smoke as evidence that independent hooks can learn specialized calculator policies. It only makes a routed diagnostic possible.
Next allowed test: Run a small task-partitioned training diagnostic that reports per-hook route counts, per-hook calculator-result accuracy, scorer calls under topk/exact assignment, and whether routed hooks interfere or specialize.
Source: `aiAgentWorkHistory/phase7/2026-05-30-left-operand-routed-multi-hook-support.md`
```
