# Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-fresh-seed.md

Summary:

- Repeated the combined sparse zero-improvement online hard memory plus additive semantic distillation source on CLI seed `7` / effective seed `9`. Source acquisition replicated: final `400/400 = 1.000`, step-800 source calc `1.000`, additive semantic token agreement `0.7403`, and memory froze after `76,800` forced evals with low controls. The trusted 600-step frozen-policy additive handoff preserved calculator accuracy (`1.000`) and low controls (`0.0250` injection-zero, `0.0325` forced-zero, `0.0225` forced-random) but reached only `0.6475` final / `0.6625` step-600 normal. A 600-step continuation improved to `0.823` final / `0.850` step-600 normal with low controls, showing usable but weaker handoff/readout geometry rather than a robust replicated pass.

Questions:

- What did we learn about Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive?
- Has Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive been tested?
- Should we repeat Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive?
- What is the status of Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive?
- What follow-up is allowed for Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-fresh-seed.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more same-op19 source-only seeds or semantic-distill weight tweaks as novelty. The source mechanism replicated; the unresolved issue is robust trusted handoff/readout behavior across seeds.

Next Allowed:

- Diagnose or improve handoff robustness, test multiple downstream handoff seeds from the same fresh source, move to streaming/fresh-prompt memory, or validate a many-calculator/routed version if it directly tests scalability.

Full Text:

```text
MIXED-POSITIVE: Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive.
Conclusion: Repeated the combined sparse zero-improvement online hard memory plus additive semantic distillation source on CLI seed `7` / effective seed `9`. Source acquisition replicated: final `400/400 = 1.000`, step-800 source calc `1.000`, additive semantic token agreement `0.7403`, and memory froze after `76,800` forced evals with low controls. The trusted 600-step frozen-policy additive handoff preserved calculator accuracy (`1.000`) and low controls (`0.0250` injection-zero, `0.0325` forced-zero, `0.0225` forced-random) but reached only `0.6475` final / `0.6625` step-600 normal. A 600-step continuation improved to `0.823` final / `0.850` step-600 normal with low controls, showing usable but weaker handoff/readout geometry rather than a robust replicated pass.
Do not repeat: Do not run more same-op19 source-only seeds or semantic-distill weight tweaks as novelty. The source mechanism replicated; the unresolved issue is robust trusted handoff/readout behavior across seeds.
Next allowed test: Diagnose or improve handoff robustness, test multiple downstream handoff seeds from the same fresh source, move to streaming/fresh-prompt memory, or validate a many-calculator/routed version if it directly tests scalability.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-fresh-seed.md`
```
