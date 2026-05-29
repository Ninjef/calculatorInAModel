# A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining.

Kind: hypothesis_memory
Status: DISPROVEN
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-decay-offramp.md

Summary:

- Decaying anchor weight `10 -> 0` over the first `200/400` unfreeze steps preserved calc accuracy at shutoff (`0.8300/0.8225`) but final calc fell to `0.5950/0.3850`, with final eval `0.5925/0.6750`.

Questions:

- What did we learn about A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining?
- Has A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining been tested?
- Should we repeat A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining?
- What is the status of A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining?
- Why did A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-decay-offramp.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Same adapted `src4_add2/src5_add5`, anchor weight `10`, decay `200`, LR `3e-4`, 400-step full unfreeze as novelty.

Next Allowed:

- Slower or floored anchor schedules, calculator-accuracy-gated unfreezing, selective unfreeze, or a source policy that is robust without anchoring.

Full Text:

```text
DISPROVEN: A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining.
Conclusion: Decaying anchor weight `10 -> 0` over the first `200/400` unfreeze steps preserved calc accuracy at shutoff (`0.8300/0.8225`) but final calc fell to `0.5950/0.3850`, with final eval `0.5925/0.6750`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, decay `200`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Slower or floored anchor schedules, calculator-accuracy-gated unfreezing, selective unfreeze, or a source policy that is robust without anchoring.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-decay-offramp.md`
```
