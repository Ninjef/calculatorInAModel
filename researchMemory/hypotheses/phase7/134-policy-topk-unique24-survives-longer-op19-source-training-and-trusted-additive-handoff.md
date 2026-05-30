# Policy-topk unique24 survives longer op19 source training and trusted additive handoff.

Kind: hypothesis_memory
Status: POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md

Summary:

- Extended the promising `topk8+unique24` sampled hard-assignment proposal from the 200-step source screen to the staged op19 `rhead64` source recipe with late recovery (`630` steps, `24/39` result classes scored per assignment). The source reached `400/400 = 1.0000` final eval and step-630 normal/calc `1.0000`, with low controls (`0.0275` injection-zero, `0.0300` forced-random). The trusted frozen-policy additive handoff from the step-630 checkpoint reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with injection-zero `0.0200` and forced-random `0.0325`. This upgrades policy-topk assignment from a short source-screen positive to a real op19 staged-transfer positive at lower assignment scoring cost, but it remains one seed/range and still uses hard assignment plus frozen transfer.

Questions:

- What did we learn about Policy-topk unique24 survives longer op19 source training and trusted additive handoff?
- Has Policy-topk unique24 survives longer op19 source training and trusted additive handoff been tested?
- Should we repeat Policy-topk unique24 survives longer op19 source training and trusted additive handoff?
- What is the status of Policy-topk unique24 survives longer op19 source training and trusted additive handoff?
- What follow-up is allowed for Policy-topk unique24 survives longer op19 source training and trusted additive handoff?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same effective-seed-43 op19 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty.

Next Allowed:

- Validate the policy-aware proposal on a fresh seed, larger operand range, many-calculator cost accounting, or with reduced prescriptiveness. Keep exact-assignment comparators and coverage/target-quality diagnostics.

Full Text:

```text
POSITIVE: Policy-topk unique24 survives longer op19 source training and trusted additive handoff.
Conclusion: Extended the promising `topk8+unique24` sampled hard-assignment proposal from the 200-step source screen to the staged op19 `rhead64` source recipe with late recovery (`630` steps, `24/39` result classes scored per assignment). The source reached `400/400 = 1.0000` final eval and step-630 normal/calc `1.0000`, with low controls (`0.0275` injection-zero, `0.0300` forced-random). The trusted frozen-policy additive handoff from the step-630 checkpoint reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with injection-zero `0.0200` and forced-random `0.0325`. This upgrades policy-topk assignment from a short source-screen positive to a real op19 staged-transfer positive at lower assignment scoring cost, but it remains one seed/range and still uses hard assignment plus frozen transfer.
Do not repeat: Do not rerun the same effective-seed-43 op19 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty.
Next allowed test: Validate the policy-aware proposal on a fresh seed, larger operand range, many-calculator cost accounting, or with reduced prescriptiveness. Keep exact-assignment comparators and coverage/target-quality diagnostics.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md`
```
