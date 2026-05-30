# Frozen result-boundary proposal critics generalize across evolving checkpoints.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-cross-checkpoint-critic-gate.md

Summary:

- Added a cross-checkpoint diagnostic that trains a sparse pairwise result-boundary critic on one checkpoint and evaluates top-8 proposal-plus-rescoring on other checkpoints from the same May 13 source lineage. Same-state top-8 recovery improved with maturity (step100 `0.48`, step400 `0.74`, step800 `0.79`), but forward transfer collapsed: train step100 to eval step400/800 recovered only `0.11/0.12`, and train step400 to eval step800 recovered `0.23`. Backward transfer from step800 was partial (`0.42` to step100, `0.58` to step400) but not strong enough. Static sparse critics are state-local, not a bridge into evolving training.

Questions:

- What did we learn about Frozen result-boundary proposal critics generalize across evolving checkpoints?
- Has Frozen result-boundary proposal critics generalize across evolving checkpoints been tested?
- Should we repeat Frozen result-boundary proposal critics generalize across evolving checkpoints?
- What is the status of Frozen result-boundary proposal critics generalize across evolving checkpoints?
- Why did Frozen result-boundary proposal critics generalize across evolving checkpoints fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-cross-checkpoint-critic-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not use a frozen/static result-boundary critic trained at one checkpoint as a scalable source-training proposal mechanism, and do not run same-state critic count/seed tweaks as evidence of evolving training viability.

Next Allowed:

- Result-boundary proposals need online refresh, state calibration, or explicit evolving validation; otherwise move to a different less-prescriptive credit-assignment family.

Full Text:

```text
DISPROVEN: Frozen result-boundary proposal critics generalize across evolving checkpoints.
Conclusion: Added a cross-checkpoint diagnostic that trains a sparse pairwise result-boundary critic on one checkpoint and evaluates top-8 proposal-plus-rescoring on other checkpoints from the same May 13 source lineage. Same-state top-8 recovery improved with maturity (step100 `0.48`, step400 `0.74`, step800 `0.79`), but forward transfer collapsed: train step100 to eval step400/800 recovered only `0.11/0.12`, and train step400 to eval step800 recovered `0.23`. Backward transfer from step800 was partial (`0.42` to step100, `0.58` to step400) but not strong enough. Static sparse critics are state-local, not a bridge into evolving training.
Do not repeat: Do not use a frozen/static result-boundary critic trained at one checkpoint as a scalable source-training proposal mechanism, and do not run same-state critic count/seed tweaks as evidence of evolving training viability.
Next allowed test: Result-boundary proposals need online refresh, state calibration, or explicit evolving validation; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-cross-checkpoint-critic-gate.md`
```
