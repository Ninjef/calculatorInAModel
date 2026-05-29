# A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md

Summary:

- After wiring adaptive recovery to switch both LR and forced-true weight, `result_policy_argmax_result_accuracy >= 0.65` with min step `500` fired at step `528`; source final eval was only `0.8250`, but the trusted 600-step frozen-policy handoff reached `0.9850` final eval / `0.9775` step-600 snapshot with learned calc `0.8425`, injection-zero `0.1325`, and forced-random `0.1325`. This beats the fixed step-600 handoff final eval (`0.9400`) but has higher controls than the fixed-step run (`0.0800`/`0.0775`).

Questions:

- What did we learn about A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen?
- Has A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen been tested?
- Should we repeat A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen?
- What is the status of A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen?
- What follow-up is allowed for A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-14 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive recovery plus 600-step handoff as novelty.

Next Allowed:

- Replicate the adaptive trigger on a fresh scheduled source, or use a smoothed/conjunctive trigger that preserves the handoff lift while reducing zero/random controls.

Full Text:

```text
MIXED-POSITIVE: A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen.
Conclusion: After wiring adaptive recovery to switch both LR and forced-true weight, `result_policy_argmax_result_accuracy >= 0.65` with min step `500` fired at step `528`; source final eval was only `0.8250`, but the trusted 600-step frozen-policy handoff reached `0.9850` final eval / `0.9775` step-600 snapshot with learned calc `0.8425`, injection-zero `0.1325`, and forced-random `0.1325`. This beats the fixed step-600 handoff final eval (`0.9400`) but has higher controls than the fixed-step run (`0.0800`/`0.0775`).
Do not repeat: The same seed-14 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Replicate the adaptive trigger on a fresh scheduled source, or use a smoothed/conjunctive trigger that preserves the handoff lift while reducing zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`
```
