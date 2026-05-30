# Simple online-calibrated result-boundary proposal critics do not restore same-state quality.

Kind: hypothesis_memory
Status: PARTIAL-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-result-boundary-online-calibrated-critic-gate.md

Summary:

- Extended the cross-checkpoint critic diagnostic with warm-start online calibration: retarget critic normalization at the eval checkpoint and fine-tune on fresh sparse forced scores before proposing top-8 candidates. Calibration repaired part of the forward-transfer collapse but did not match same-state quality. Step400-to-step800 top-8 recovery improved from frozen `0.23` to adapted `0.59` with `2` fresh scores per train prompt, `0.54` with `4`, and `0.62` with `8`; the same-state step800 critic was `0.79`. Step100-to-step400/800 with `2` fresh scores improved only to `0.36`/`0.41`. Simple warm-start calibration is helpful, but not enough to be a scalable source-training proposal mechanism.

Questions:

- What did we learn about Simple online-calibrated result-boundary proposal critics do not restore same-state quality?
- Has Simple online-calibrated result-boundary proposal critics do not restore same-state quality been tested?
- Should we repeat Simple online-calibrated result-boundary proposal critics do not restore same-state quality?
- What is the status of Simple online-calibrated result-boundary proposal critics do not restore same-state quality?
- Why did Simple online-calibrated result-boundary proposal critics do not restore same-state quality fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-online-calibrated-critic-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not wire this warm-start calibrated critic into source training or spend mainline compute on small adapt-lr/epoch/sample-count tweaks as if the proposal mechanism were solved.

Next Allowed:

- Only continue result-boundary proposals with a stronger online learner, active proposal/training co-design, or a materially different state-calibrated objective; otherwise move to a different less-prescriptive credit-assignment family.

Full Text:

```text
PARTIAL-NEGATIVE: Simple online-calibrated result-boundary proposal critics do not restore same-state quality.
Conclusion: Extended the cross-checkpoint critic diagnostic with warm-start online calibration: retarget critic normalization at the eval checkpoint and fine-tune on fresh sparse forced scores before proposing top-8 candidates. Calibration repaired part of the forward-transfer collapse but did not match same-state quality. Step400-to-step800 top-8 recovery improved from frozen `0.23` to adapted `0.59` with `2` fresh scores per train prompt, `0.54` with `4`, and `0.62` with `8`; the same-state step800 critic was `0.79`. Step100-to-step400/800 with `2` fresh scores improved only to `0.36`/`0.41`. Simple warm-start calibration is helpful, but not enough to be a scalable source-training proposal mechanism.
Do not repeat: Do not wire this warm-start calibrated critic into source training or spend mainline compute on small adapt-lr/epoch/sample-count tweaks as if the proposal mechanism were solved.
Next allowed test: Only continue result-boundary proposals with a stronger online learner, active proposal/training co-design, or a materially different state-calibrated objective; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-online-calibrated-critic-gate.md`
```
