# Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates.

Kind: hypothesis_memory
Status: PARTIAL
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-01-target-stratified-prior-fit-gate.md

Summary:

- Changing only the prior fit sampler from random half-memory to target-stratified half-memory reached source overall `0.9900`, heldout `0.9375`, low heldout controls, and trusted frozen-policy handoff `0.9975` with diagnostic calc `1.0000` and low final counterfactuals. This reverses the random-half failure and cuts forced evals to `67,584`, but prior updates stayed at `2501`.

Questions:

- What did we learn about Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates?
- Has Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates been tested?
- Should we repeat Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates?
- What is the status of Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-prior-fit-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Random prior fit-batch-size ladders, or same target-stratified op19 seed as novelty.

Next Allowed:

- Combine target-stratified sampling with convergence/validation stopping, or stress it on a fresh seed/range axis before promoting it to default.

Full Text:

```text
PARTIAL: Target-stratified half-memory prior fitting preserves the integrated numeric-prior source and handoff gates.
Conclusion: Changing only the prior fit sampler from random half-memory to target-stratified half-memory reached source overall `0.9900`, heldout `0.9375`, low heldout controls, and trusted frozen-policy handoff `0.9975` with diagnostic calc `1.0000` and low final counterfactuals. This reverses the random-half failure and cuts forced evals to `67,584`, but prior updates stayed at `2501`.
Do not repeat: Random prior fit-batch-size ladders, or same target-stratified op19 seed as novelty.
Next allowed test: Combine target-stratified sampling with convergence/validation stopping, or stress it on a fresh seed/range axis before promoting it to default.
Source: `aiAgentWorkHistory/phase7/2026-06-01-target-stratified-prior-fit-gate.md`
```
