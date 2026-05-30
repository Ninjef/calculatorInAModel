# Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-forced-margin-low-lr-source-recovery.md

Summary:

- Continuing the longer one-negative forced-margin step-600 source checkpoint for `30` low-LR CPU steps (`lr=0.0003`, margin weight reduced from `0.5` to `0.1`, source stabilization retained) raised source calculator accuracy from `0.5225` to `0.7725` and final source eval to `0.7825`. The trusted frozen-policy 600-step non-bottleneck handoff from recovered step `30` reached `0.8700` final eval / `0.9050` step-600 normal, with injection-zero `0.0000`, forced-random `0.0313`, and learned calculator accuracy `0.8594`. This beats the unrecovered forced-margin handoffs (`0.7330-0.7400` final) and the old scheduled forced-true step-600 handoff (`0.7725` final), but remains below automated scheduled-source recovery (`0.9400` final) and still depends on hard assignment plus true-result contrastive forcing.

Questions:

- What did we learn about Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited?
- Has Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited been tested?
- Should we repeat Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited?
- What is the status of Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited?
- Why did Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited fail?
- What follow-up is allowed for Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-forced-margin-low-lr-source-recovery.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same seed-15 step-600 forced-margin checkpoint recovery with `lr=3e-4`, margin weight `0.1`, `30` source steps, and the same 600-step frozen-policy handoff as novelty.

Next Allowed:

- If staying in forced-margin, test fresh-seed stability or fold the recovery into an automated source run. Otherwise use it as evidence that source objectives need late gentle recovery while moving toward less-prescriptive target construction or estimator work.

Full Text:

```text
POSITIVE: Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited.
Conclusion: Continuing the longer one-negative forced-margin step-600 source checkpoint for `30` low-LR CPU steps (`lr=0.0003`, margin weight reduced from `0.5` to `0.1`, source stabilization retained) raised source calculator accuracy from `0.5225` to `0.7725` and final source eval to `0.7825`. The trusted frozen-policy 600-step non-bottleneck handoff from recovered step `30` reached `0.8700` final eval / `0.9050` step-600 normal, with injection-zero `0.0000`, forced-random `0.0313`, and learned calculator accuracy `0.8594`. This beats the unrecovered forced-margin handoffs (`0.7330-0.7400` final) and the old scheduled forced-true step-600 handoff (`0.7725` final), but remains below automated scheduled-source recovery (`0.9400` final) and still depends on hard assignment plus true-result contrastive forcing.
Do not repeat: Do not rerun the same seed-15 step-600 forced-margin checkpoint recovery with `lr=3e-4`, margin weight `0.1`, `30` source steps, and the same 600-step frozen-policy handoff as novelty.
Next allowed test: If staying in forced-margin, test fresh-seed stability or fold the recovery into an automated source run. Otherwise use it as evidence that source objectives need late gentle recovery while moving toward less-prescriptive target construction or estimator work.
Source: `aiAgentWorkHistory/phase7/2026-05-29-forced-margin-low-lr-source-recovery.md`
```
