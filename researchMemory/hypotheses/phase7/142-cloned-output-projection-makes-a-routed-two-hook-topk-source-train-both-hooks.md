# Cloned output projection makes a routed two-hook topk source train both hooks.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-routed-cloned-output-source-gate.md

Summary:

- Made result-policy reads route-aware, added per-route assignment metrics and forced-eval counts, and tested op19 `rhead64` two-hook `left_operand_mod` routing. Without cloning the primary calculator output projection into extra hooks, routed exact/topk 200-step source runs mostly trained hook 0 only: exact reached final eval `0.4825` with hook calc `0.8767/0.0387`, and topk8+unique24 reached `0.5250` with hook calc `0.9315/0.0110`. A 50-step exact diagnostic showed why: hook 1 received targets, but target accuracy was only `0.0839`, because the extra hook's frozen random output projection made forced-result scoring semantically invalid. Adding `--clone-primary-calculator-output-proj` fixed the semantic interface: exact50 route target accuracy became `0.8831/0.9333`, oracle eval became `1.0000`, and the cloned topk8+unique24 source200 reached final eval `361/400 = 0.9025`, step-200 normal `0.9250`, hook calc `0.9315/0.9171`, target accuracy `1.0000`, and scored only `24/39` result classes (`9,600` forced evals per full-grid step versus `15,600` exact). This is the first fair routed two-hook source positive, but it is still prescriptive/source-only and injection-zero was high (`0.4325`), so it does not yet prove causal non-bottleneck use.

Questions:

- What did we learn about Cloned output projection makes a routed two-hook topk source train both hooks?
- Has Cloned output projection makes a routed two-hook topk source train both hooks been tested?
- Should we repeat Cloned output projection makes a routed two-hook topk source train both hooks?
- What is the status of Cloned output projection makes a routed two-hook topk source train both hooks?
- What follow-up is allowed for Cloned output projection makes a routed two-hook topk source train both hooks?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-routed-cloned-output-source-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run routed multi-hook source diagnostics with frozen random extra-hook output projections and interpret hook 1 collapse as a routing/assignment failure. The output semantic interface must be cloned/shared or trained before assignment targets are meaningful.

Next Allowed:

- Run trusted additive handoff from the cloned routed topk checkpoint, validate on a fresh seed, or replace cloned initialization with a genuinely shared/tied output projection that reduces many-calculator parameters.

Full Text:

```text
MIXED-POSITIVE: Cloned output projection makes a routed two-hook topk source train both hooks.
Conclusion: Made result-policy reads route-aware, added per-route assignment metrics and forced-eval counts, and tested op19 `rhead64` two-hook `left_operand_mod` routing. Without cloning the primary calculator output projection into extra hooks, routed exact/topk 200-step source runs mostly trained hook 0 only: exact reached final eval `0.4825` with hook calc `0.8767/0.0387`, and topk8+unique24 reached `0.5250` with hook calc `0.9315/0.0110`. A 50-step exact diagnostic showed why: hook 1 received targets, but target accuracy was only `0.0839`, because the extra hook's frozen random output projection made forced-result scoring semantically invalid. Adding `--clone-primary-calculator-output-proj` fixed the semantic interface: exact50 route target accuracy became `0.8831/0.9333`, oracle eval became `1.0000`, and the cloned topk8+unique24 source200 reached final eval `361/400 = 0.9025`, step-200 normal `0.9250`, hook calc `0.9315/0.9171`, target accuracy `1.0000`, and scored only `24/39` result classes (`9,600` forced evals per full-grid step versus `15,600` exact). This is the first fair routed two-hook source positive, but it is still prescriptive/source-only and injection-zero was high (`0.4325`), so it does not yet prove causal non-bottleneck use.
Do not repeat: Do not run routed multi-hook source diagnostics with frozen random extra-hook output projections and interpret hook 1 collapse as a routing/assignment failure. The output semantic interface must be cloned/shared or trained before assignment targets are meaningful.
Next allowed test: Run trusted additive handoff from the cloned routed topk checkpoint, validate on a fresh seed, or replace cloned initialization with a genuinely shared/tied output projection that reduces many-calculator parameters.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-cloned-output-source-gate.md`
```
