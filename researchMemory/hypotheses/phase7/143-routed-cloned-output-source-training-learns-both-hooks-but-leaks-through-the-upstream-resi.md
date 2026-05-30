# Routed cloned-output source training learns both hooks but leaks through the upstream residual.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md

Summary:

- Tightened `--freeze-semantic-decoder` so it also freezes calculator output projections for `ste` handoff runs, then tested cloned-output routed handoff/source controls. A strict frozen-policy handoff from the 200-step cloned routed source reached high additive accuracy (`0.9075` final, `0.9175` step-600 normal) but failed the causal control (`0.4925` injection-zero, `0.0175` forced-random). The matched `embd32` routed source630, using the same product-decoder parity checkpoint and architecture as the single-hook positive, reached `400/400 = 1.0000` final and step-630 normal `0.9975` with both hooks trained (`1.0000/0.9944` hook calc), but still had high injection-zero (`0.4600`, 128-sample final counterfactual `0.53125`). A frozen-upstream routed source200 reduced leakage (`0.1875` injection-zero) but learned much more slowly (`0.4150` normal, hook calc `0.4384/0.3867`). Therefore routed multi-hook sparse assignment can train active hooks, but open-upstream source acquisition creates a direct residual path, and freezing upstream trades leakage for undertraining at this budget.

Questions:

- What did we learn about Routed cloned-output source training learns both hooks but leaks through the upstream residual?
- Has Routed cloned-output source training learns both hooks but leaks through the upstream residual been tested?
- Should we repeat Routed cloned-output source training learns both hooks but leaks through the upstream residual?
- What is the status of Routed cloned-output source training learns both hooks but leaks through the upstream residual?
- Why did Routed cloned-output source training learns both hooks but leaks through the upstream residual fail?
- What follow-up is allowed for Routed cloned-output source training learns both hooks but leaks through the upstream residual?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat high normal accuracy in open-upstream routed source/handoff runs as causal calculator use when injection-zero is around `0.46-0.53`. Do not rerun the same 200-step handoff from the leaked source as evidence.

Next Allowed:

- Develop an anti-leak routed source recipe: longer frozen-upstream training, partial/upstream trust-region freezing, explicit causal-gap/source ablation pressure, or a shared/tied output-projection design with a source control gate before any handoff.

Full Text:

```text
MIXED-NEGATIVE: Routed cloned-output source training learns both hooks but leaks through the upstream residual.
Conclusion: Tightened `--freeze-semantic-decoder` so it also freezes calculator output projections for `ste` handoff runs, then tested cloned-output routed handoff/source controls. A strict frozen-policy handoff from the 200-step cloned routed source reached high additive accuracy (`0.9075` final, `0.9175` step-600 normal) but failed the causal control (`0.4925` injection-zero, `0.0175` forced-random). The matched `embd32` routed source630, using the same product-decoder parity checkpoint and architecture as the single-hook positive, reached `400/400 = 1.0000` final and step-630 normal `0.9975` with both hooks trained (`1.0000/0.9944` hook calc), but still had high injection-zero (`0.4600`, 128-sample final counterfactual `0.53125`). A frozen-upstream routed source200 reduced leakage (`0.1875` injection-zero) but learned much more slowly (`0.4150` normal, hook calc `0.4384/0.3867`). Therefore routed multi-hook sparse assignment can train active hooks, but open-upstream source acquisition creates a direct residual path, and freezing upstream trades leakage for undertraining at this budget.
Do not repeat: Do not treat high normal accuracy in open-upstream routed source/handoff runs as causal calculator use when injection-zero is around `0.46-0.53`. Do not rerun the same 200-step handoff from the leaked source as evidence.
Next allowed test: Develop an anti-leak routed source recipe: longer frozen-upstream training, partial/upstream trust-region freezing, explicit causal-gap/source ablation pressure, or a shared/tied output-projection design with a source control gate before any handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`
```
