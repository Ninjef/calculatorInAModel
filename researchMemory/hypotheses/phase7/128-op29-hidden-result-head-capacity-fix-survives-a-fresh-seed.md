# op29 hidden result-head capacity fix survives a fresh seed.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-op29-rhead64-fresh-seed-replication.md

Summary:

- Repeating the op29 `rhead64` source-plus-handoff on a new CLI seed `31` / effective model seed `33` replicated the capacity fix. The source recovered from `0.7122` source calc at step `600` to `0.9967` at step `630`, with source final eval `897/900 = 0.9967`; controls stayed low at step `630` (`0.0200` injection-zero, `0.0133` forced-random). The trusted frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0344` injection-zero, `0.0111` forced-random) and learned calc `1.0000` at step `600`. This upgrades the hidden-head op29 result from a one-seed capacity rescue to a replicated staged range-capacity finding, while preserving the caveat that the method is still full-grid, prescriptive, and per-calculator-head-costly.

Questions:

- What did we learn about op29 hidden result-head capacity fix survives a fresh seed?
- Has op29 hidden result-head capacity fix survives a fresh seed been tested?
- Should we repeat op29 hidden result-head capacity fix survives a fresh seed?
- What is the status of op29 hidden result-head capacity fix survives a fresh seed?
- What follow-up is allowed for op29 hidden result-head capacity fix survives a fresh seed?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-op29-rhead64-fresh-seed-replication.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the completed op29 `rhead64` effective-seed-29 or effective-seed-33 source-plus-handoff pairs as novelty.

Next Allowed:

- Further work should either stress a new axis such as larger operand ranges or many-calculator parameter/training cost, or reduce/remove full-grid hard assignment and true-result forced-margin pressure.

Full Text:

```text
POSITIVE: op29 hidden result-head capacity fix survives a fresh seed.
Conclusion: Repeating the op29 `rhead64` source-plus-handoff on a new CLI seed `31` / effective model seed `33` replicated the capacity fix. The source recovered from `0.7122` source calc at step `600` to `0.9967` at step `630`, with source final eval `897/900 = 0.9967`; controls stayed low at step `630` (`0.0200` injection-zero, `0.0133` forced-random). The trusted frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0344` injection-zero, `0.0111` forced-random) and learned calc `1.0000` at step `600`. This upgrades the hidden-head op29 result from a one-seed capacity rescue to a replicated staged range-capacity finding, while preserving the caveat that the method is still full-grid, prescriptive, and per-calculator-head-costly.
Do not repeat: Do not rerun the completed op29 `rhead64` effective-seed-29 or effective-seed-33 source-plus-handoff pairs as novelty.
Next allowed test: Further work should either stress a new axis such as larger operand ranges or many-calculator parameter/training cost, or reduce/remove full-grid hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-rhead64-fresh-seed-replication.md`
```
