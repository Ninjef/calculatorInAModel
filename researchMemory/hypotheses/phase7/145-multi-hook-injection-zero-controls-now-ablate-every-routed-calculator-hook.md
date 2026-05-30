# Multi-hook injection-zero controls now ablate every routed calculator hook.

Kind: hypothesis_memory
Status: POSITIVE-CORRECTION
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-multihook-injection-zero-fix.md

Summary:

- Fixed `temporary_calculator_injection_scale` so evaluation, causal-gap, and zero-injection contexts set `injection_scale` on every module returned by `model.calculator_hook_modules()`, not just `model.calculator_hook`. Added a regression test that two routed hooks are both scaled inside the context and restored afterward. Corrected evidence changed the routed interpretation: a matched source200 rerun reached `0.9400` final / `0.9225` snapshot normal with low controls (`0.0200` injection-zero, `0.0325` forced-random) and both hooks trained (`0.9406/0.9006` hook calc). Re-evaluating the previous open-upstream source630 checkpoint gave `1.0000` final / `0.9950` snapshot normal, `0.0250` injection-zero, and hook calc `1.0000/0.9893`. Re-evaluating the strict source200 handoff checkpoint gave `0.9075` final / `0.9250` snapshot normal, `0.0000` injection-zero, `0.0300` forced-random, and hook calc `0.9108/0.9198`. The routed multi-hook source and handoff were calculator-causal under corrected controls.

Questions:

- What did we learn about Multi-hook injection-zero controls now ablate every routed calculator hook?
- Has Multi-hook injection-zero controls now ablate every routed calculator hook been tested?
- Should we repeat Multi-hook injection-zero controls now ablate every routed calculator hook?
- What is the status of Multi-hook injection-zero controls now ablate every routed calculator hook?
- What follow-up is allowed for Multi-hook injection-zero controls now ablate every routed calculator hook?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-multihook-injection-zero-fix.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not interpret old routed multi-hook `injection_zero_exact_match` numbers from before this fix as causal evidence. Any multi-hook ablation or causal-gap objective must verify all hook scales are changed.

Next Allowed:

- Use corrected controls to validate the stronger `embd32` routed source630 in additive handoff, test fresh seeds or more routed hooks, and replace cloned output projections with shared/tied projections for real many-calculator parameter scaling.

Full Text:

```text
POSITIVE-CORRECTION: Multi-hook injection-zero controls now ablate every routed calculator hook.
Conclusion: Fixed `temporary_calculator_injection_scale` so evaluation, causal-gap, and zero-injection contexts set `injection_scale` on every module returned by `model.calculator_hook_modules()`, not just `model.calculator_hook`. Added a regression test that two routed hooks are both scaled inside the context and restored afterward. Corrected evidence changed the routed interpretation: a matched source200 rerun reached `0.9400` final / `0.9225` snapshot normal with low controls (`0.0200` injection-zero, `0.0325` forced-random) and both hooks trained (`0.9406/0.9006` hook calc). Re-evaluating the previous open-upstream source630 checkpoint gave `1.0000` final / `0.9950` snapshot normal, `0.0250` injection-zero, and hook calc `1.0000/0.9893`. Re-evaluating the strict source200 handoff checkpoint gave `0.9075` final / `0.9250` snapshot normal, `0.0000` injection-zero, `0.0300` forced-random, and hook calc `0.9108/0.9198`. The routed multi-hook source and handoff were calculator-causal under corrected controls.
Do not repeat: Do not interpret old routed multi-hook `injection_zero_exact_match` numbers from before this fix as causal evidence. Any multi-hook ablation or causal-gap objective must verify all hook scales are changed.
Next allowed test: Use corrected controls to validate the stronger `embd32` routed source630 in additive handoff, test fresh seeds or more routed hooks, and replace cloned output projections with shared/tied projections for real many-calculator parameter scaling.
Source: `aiAgentWorkHistory/phase7/2026-05-30-multihook-injection-zero-fix.md`
```
