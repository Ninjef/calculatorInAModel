# Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md

Summary:

- Added `--result-policy-improvement-assignment-sample-count` to approximate hard improvement assignment by scoring the learned result plus uniform random result classes, then ran an op19 `rhead64` 200-step source gate against an exact full-result ceiling. The exact branch scored all `39` result classes, reached best snapshot normal `0.8625` at step `150`, final eval `294/400 = 0.7350`, step-200 true-result coverage `1.0000`, and assignment target accuracy `0.9900`. Sample16 scored `16/39` classes but reached only best snapshot `0.3650`, final `141/400 = 0.3525`, true coverage `0.6125`, and target accuracy `0.4581`. Sample32 scored `32/39` classes but reached only best snapshot `0.4050`, final `152/400 = 0.3800`, true coverage `0.7400`, and target accuracy `0.6773`. Wall-clock savings were modest at this local op19 gate (about `115s` exact, `88s` sample16, `106s` sample32), so the accuracy loss is not a good trade.

Questions:

- What did we learn about Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling?
- Has Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling been tested?
- Should we repeat Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling?
- What is the status of Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling?
- Why did Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more uniform sample-count ladders on the same op19 `rhead64` 200-step forced-margin source gate as novelty, and do not expect duplicate-prone uniform result sampling to solve hard-assignment cost without a proposal or coverage mechanism.

Next Allowed:

- Assignment-cost reduction needs a smarter candidate mechanism, such as coverage-aware/active/structured proposals, an accumulated candidate state validated beyond prompt transduction, or a different non-enumerative credit signal. Compare any such method to an exact-grid assignment ceiling.

Full Text:

```text
DISPROVEN: Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling.
Conclusion: Added `--result-policy-improvement-assignment-sample-count` to approximate hard improvement assignment by scoring the learned result plus uniform random result classes, then ran an op19 `rhead64` 200-step source gate against an exact full-result ceiling. The exact branch scored all `39` result classes, reached best snapshot normal `0.8625` at step `150`, final eval `294/400 = 0.7350`, step-200 true-result coverage `1.0000`, and assignment target accuracy `0.9900`. Sample16 scored `16/39` classes but reached only best snapshot `0.3650`, final `141/400 = 0.3525`, true coverage `0.6125`, and target accuracy `0.4581`. Sample32 scored `32/39` classes but reached only best snapshot `0.4050`, final `152/400 = 0.3800`, true coverage `0.7400`, and target accuracy `0.6773`. Wall-clock savings were modest at this local op19 gate (about `115s` exact, `88s` sample16, `106s` sample32), so the accuracy loss is not a good trade.
Do not repeat: Do not run more uniform sample-count ladders on the same op19 `rhead64` 200-step forced-margin source gate as novelty, and do not expect duplicate-prone uniform result sampling to solve hard-assignment cost without a proposal or coverage mechanism.
Next allowed test: Assignment-cost reduction needs a smarter candidate mechanism, such as coverage-aware/active/structured proposals, an accumulated candidate state validated beyond prompt transduction, or a different non-enumerative credit signal. Compare any such method to an exact-grid assignment ceiling.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md`
```
