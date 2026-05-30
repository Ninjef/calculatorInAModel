# Routed cloned-output source/handoff controls were only zeroing the primary hook.

Kind: hypothesis_memory
Status: SUPERSEDED-MEASUREMENT-BUG
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md

Summary:

- Tightened `--freeze-semantic-decoder` so it also freezes calculator output projections for `ste` handoff runs, then tested cloned-output routed handoff/source controls. A strict frozen-policy handoff from the 200-step cloned routed source reached high additive accuracy (`0.9075` final, `0.9175` step-600 normal) but failed the causal control (`0.4925` injection-zero, `0.0175` forced-random). The matched `embd32` routed source630, using the same product-decoder parity checkpoint and architecture as the single-hook positive, reached `400/400 = 1.0000` final and step-630 normal `0.9975` with both hooks trained (`1.0000/0.9944` hook calc), but still had high injection-zero (`0.4600`, 128-sample final counterfactual `0.53125`). A frozen-upstream routed source200 reduced leakage (`0.1875` injection-zero) but learned much more slowly (`0.4150` normal, hook calc `0.4384/0.3867`). Therefore routed multi-hook sparse assignment can train active hooks, but open-upstream source acquisition creates a direct residual path, and freezing upstream trades leakage for undertraining at this budget.

Questions:

- What did we learn about Routed cloned-output source/handoff controls were only zeroing the primary hook?
- Has Routed cloned-output source/handoff controls were only zeroing the primary hook been tested?
- Should we repeat Routed cloned-output source/handoff controls were only zeroing the primary hook?
- What is the status of Routed cloned-output source/handoff controls were only zeroing the primary hook?
- Why did Routed cloned-output source/handoff controls were only zeroing the primary hook fail?
- What follow-up is allowed for Routed cloned-output source/handoff controls were only zeroing the primary hook?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not cite the old routed multi-hook injection-zero values unless explicitly labeling them invalid. Multi-hook counterfactuals must scale every calculator hook, not only the primary hook.

Next Allowed:

- With corrected controls, validate the stronger `embd32` routed source630 through a trusted additive handoff, replicate routed training on a fresh seed/more hooks, or replace cloned output projections with shared/tied output projections for parameter scalability.

Full Text:

```text
SUPERSEDED-MEASUREMENT-BUG: Routed cloned-output source/handoff controls were only zeroing the primary hook.
Conclusion: Tightened `--freeze-semantic-decoder` so it also freezes calculator output projections for `ste` handoff runs, then tested cloned-output routed handoff/source controls. A strict frozen-policy handoff from the 200-step cloned routed source reached high additive accuracy (`0.9075` final, `0.9175` step-600 normal) but failed the causal control (`0.4925` injection-zero, `0.0175` forced-random). The matched `embd32` routed source630, using the same product-decoder parity checkpoint and architecture as the single-hook positive, reached `400/400 = 1.0000` final and step-630 normal `0.9975` with both hooks trained (`1.0000/0.9944` hook calc), but still had high injection-zero (`0.4600`, 128-sample final counterfactual `0.53125`). A frozen-upstream routed source200 reduced leakage (`0.1875` injection-zero) but learned much more slowly (`0.4150` normal, hook calc `0.4384/0.3867`). Therefore routed multi-hook sparse assignment can train active hooks, but open-upstream source acquisition creates a direct residual path, and freezing upstream trades leakage for undertraining at this budget.
Correction: the apparent `0.46-0.53` leakage was a multi-hook control bug. `temporary_calculator_injection_scale` only changed `model.calculator_hook`, leaving `extra_calculator_hooks` active. After fixing it to scale all hook modules, the same open-upstream source630 checkpoint re-evaluated at `1.0000` final / `0.9950` snapshot normal with `0.0250` injection-zero, and the strict source200 handoff checkpoint re-evaluated at `0.9075` final / `0.9250` snapshot normal with `0.0000` injection-zero. The source/handoff were calculator-causal; the old leakage interpretation is superseded.
Do not repeat: Do not cite the old routed multi-hook injection-zero values unless explicitly labeling them invalid. Multi-hook counterfactuals must scale every calculator hook, not only the primary hook.
Next allowed test: With corrected controls, validate the stronger `embd32` routed source630 through a trusted additive handoff, replicate routed training on a fresh seed/more hooks, or replace cloned output projections with shared/tied output projections for parameter scalability.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`
```
