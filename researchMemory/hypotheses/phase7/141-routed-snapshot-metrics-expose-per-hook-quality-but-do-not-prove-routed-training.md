# Routed snapshot metrics expose per-hook quality but do not prove routed training.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-routed-multi-hook-snapshot-metrics.md

Summary:

- Updated diagnostic snapshots so routed runs select each example's active hook trace instead of always reading the primary hook. Snapshot rows now include `calculator_hook_route_distribution`, `calculator_hook_active_count`, and per-hook fields such as `hook_0_route_count`, `hook_0_normal_exact_match`, `hook_0_operand_exact_match`, `hook_0_calculator_result_accuracy`, and `hook_0_mean_sampled_logp`. A zero-step smoke with `--calculator-hook-count 2 --calculator-hook-routing left_operand_mod --snapshot-every 1` wrote balanced route counts (`{"0": 4, "1": 4}`) and per-hook accuracy columns to `diagnostic_snapshots.csv`; regression tests passed (`141 passed`).

Questions:

- What did we learn about Routed snapshot metrics expose per-hook quality but do not prove routed training?
- Has Routed snapshot metrics expose per-hook quality but do not prove routed training been tested?
- Should we repeat Routed snapshot metrics expose per-hook quality but do not prove routed training?
- What is the status of Routed snapshot metrics expose per-hook quality but do not prove routed training?
- What follow-up is allowed for Routed snapshot metrics expose per-hook quality but do not prove routed training?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-routed-multi-hook-snapshot-metrics.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat routed snapshot fields as training evidence. They are instrumentation for measuring specialization/interference in the next routed training diagnostic.

Next Allowed:

- Run the small task-partitioned training diagnostic promised by the routing support: compare exact/topk scorer calls, route balance, per-hook calculator-result accuracy, and normal/injection-zero controls over training.

Full Text:

```text
PARTIAL: Routed snapshot metrics expose per-hook quality but do not prove routed training.
Conclusion: Updated diagnostic snapshots so routed runs select each example's active hook trace instead of always reading the primary hook. Snapshot rows now include `calculator_hook_route_distribution`, `calculator_hook_active_count`, and per-hook fields such as `hook_0_route_count`, `hook_0_normal_exact_match`, `hook_0_operand_exact_match`, `hook_0_calculator_result_accuracy`, and `hook_0_mean_sampled_logp`. A zero-step smoke with `--calculator-hook-count 2 --calculator-hook-routing left_operand_mod --snapshot-every 1` wrote balanced route counts (`{"0": 4, "1": 4}`) and per-hook accuracy columns to `diagnostic_snapshots.csv`; regression tests passed (`141 passed`).
Do not repeat: Do not treat routed snapshot fields as training evidence. They are instrumentation for measuring specialization/interference in the next routed training diagnostic.
Next allowed test: Run the small task-partitioned training diagnostic promised by the routing support: compare exact/topk scorer calls, route balance, per-hook calculator-result accuracy, and normal/injection-zero controls over training.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-multi-hook-snapshot-metrics.md`
```
